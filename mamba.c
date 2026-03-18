/* Inference for Mamba-3 model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Mamba-3 model
//
// Key innovations over Mamba-2 (ICLR 2026, arXiv:2603.15569):
//   1) Trapezoidal discretization (Prop. 1, Eq. 4):
//        h_t = α_t * h_{t-1} + β_t * B̄_{t-1}x_{t-1} + γ_t * B̄_t * x_t
//   2) Data-dependent RoPE on B, C (complex SSM via rotation, Prop. 3-4)
//   3) QK-normalization on B and C (RMSNorm, replaces post-SSD norm)
//   4) Learnable BC bias (head-specific, initialized to ones, Appendix G)
//   5) No short convolution (trapezoidal + bias makes conv1d unnecessary)
//   6) Llama-style architecture: RMSNorm → SSM → residual → RMSNorm → SwiGLU MLP → residual

typedef struct {
    int n_layers;    // number of layers (each = SSM mixer + SwiGLU MLP)
    int vocab_size;  // vocabulary size
    int dim;         // model dimension (D)
    int d_inner;     // inner dimension (expand * dim, E=2 typically)
    int d_state;     // SSM state dimension (N), must be even for RoPE
    int headdim;     // head dimension (P)
    int nheads;      // number of heads = d_inner / headdim
    int d_mlp_inner; // SwiGLU MLP inner dimension (~8/3 * dim, rounded to 256)
    int shared_classifier;
    int rounded_vocab_size;
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table; // (rounded_vocab_size, dim)
    // SSM mixer weights (per layer)
    float* mixer_norm; // (layer, dim) — pre-norm before SSM mixer
    // Mamba3: in_proj projects to (z, x, B, C, dt, lam, theta)
    //   z:     d_inner
    //   x:     d_inner
    //   B:     d_state  (shared across heads before bias broadcast)
    //   C:     d_state
    //   dt:    nheads
    //   lam:   nheads   (trapezoidal interpolation parameter λ)
    //   theta: d_state/2 (data-dependent RoPE angles)
    // d_in_proj = 2*d_inner + 2*d_state + 2*nheads + d_state/2
    float* in_proj;    // (layer, d_in_proj, dim)
    // SSM parameters (A pre-converted: A = -exp(A_log), so always negative)
    float* A;          // (layer, nheads)
    float* D;          // (layer, nheads)  skip connection
    float* dt_bias;    // (layer, nheads)
    // QK-Normalization weights for B and C (Section 3.4)
    float* B_norm;     // (layer, d_state)
    float* C_norm;     // (layer, d_state)
    // BC bias: head-specific, channel-wise, initialized to ones (Appendix G)
    float* B_bias;     // (layer, nheads, d_state)
    float* C_bias;     // (layer, nheads, d_state)
    // output projection
    float* out_proj;   // (layer, dim, d_inner)
    // SwiGLU MLP weights (per layer, Llama-style)
    float* mlp_norm;   // (layer, dim) — pre-norm before MLP
    float* w_gate;     // (layer, d_mlp_inner, dim)
    float* w_up;       // (layer, d_mlp_inner, dim)
    float* w_down;     // (layer, dim, d_mlp_inner)
    // final norm
    float* final_norm; // (dim)
    // LM head (may be weight-tied with token_embedding_table)
    float* lm_head;    // (rounded_vocab_size, dim)
} MambaWeights;

typedef struct {
    // scratch buffers reused across layers
    float* input;       // (dim)        residual stream
    float* hidden_state;// (dim)        scratch: normed input or matmul output
    float* proj;        // (d_in_proj)  in_proj output
    float* x;           // (d_inner)    SSM input values, reshaped as (nheads, headdim)
    float* B_heads;     // (nheads, d_state)  B after QK-norm + bias + RoPE
    float* C_heads;     // (nheads, d_state)  C after QK-norm + bias + RoPE
    float* dt;          // (nheads)     step sizes after softplus
    float* Bx;          // (nheads, headdim, d_state)  current B⊗x outer product
    float* y;           // (d_inner)    SSM output before gating
    float* mlp_gate;    // (d_mlp_inner)
    float* mlp_up;      // (d_mlp_inner)
    float* logits;      // (rounded_vocab_size)
    // persistent inference state (separate allocation per layer)
    float* ssm_state;   // (n_layers, nheads, headdim, d_state)
    float* prev_Bx;     // (n_layers, nheads, headdim, d_state)  β term from previous step
    float* cum_angle;   // (n_layers, nheads, d_state/2)  cumulative RoPE angles Σ Δ_i*θ_i
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    MambaWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Mamba;

void malloc_run_state(RunState* s, Config* p) {
    int d_state_half = p->d_state / 2;
    int d_in_proj = 2 * p->d_inner + 2 * p->d_state + 2 * p->nheads + d_state_half;

    s->input        = malloc(p->dim * sizeof(float));
    s->hidden_state = malloc(p->dim * sizeof(float));
    s->proj         = malloc(d_in_proj * sizeof(float));
    s->x            = malloc(p->d_inner * sizeof(float));
    s->B_heads      = malloc(p->nheads * p->d_state * sizeof(float));
    s->C_heads      = malloc(p->nheads * p->d_state * sizeof(float));
    s->dt           = malloc(p->nheads * sizeof(float));
    s->Bx           = malloc(p->nheads * p->headdim * p->d_state * sizeof(float));
    s->y            = malloc(p->d_inner * sizeof(float));
    s->mlp_gate     = malloc(p->d_mlp_inner * sizeof(float));
    s->mlp_up       = malloc(p->d_mlp_inner * sizeof(float));
    s->logits       = malloc(p->rounded_vocab_size * sizeof(float));

    s->ssm_state = calloc(p->n_layers * p->nheads * p->headdim * p->d_state, sizeof(float));
    s->prev_Bx   = calloc(p->n_layers * p->nheads * p->headdim * p->d_state, sizeof(float));
    s->cum_angle = calloc(p->n_layers * p->nheads * d_state_half, sizeof(float));

    if (!s->input || !s->hidden_state || !s->proj || !s->x
     || !s->B_heads || !s->C_heads || !s->dt || !s->Bx || !s->y
     || !s->mlp_gate || !s->mlp_up || !s->logits
     || !s->ssm_state || !s->prev_Bx || !s->cum_angle) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void reset_internal_state(Mamba* mamba) {
    RunState* s = &mamba->state;
    Config* p = &mamba->config;
    int d_state_half = p->d_state / 2;
    memset(s->ssm_state, 0, p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float));
    memset(s->prev_Bx,   0, p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float));
    memset(s->cum_angle, 0, p->n_layers * p->nheads * d_state_half * sizeof(float));
}

char* get_internal_state(Mamba* mamba, int* state_size) {
    Config* p = &mamba->config;
    RunState* s = &mamba->state;
    int d_state_half = p->d_state / 2;
    unsigned int ssm_size = p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float);
    unsigned int pbx_size = p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float);
    unsigned int ang_size = p->n_layers * p->nheads * d_state_half * sizeof(float);
    unsigned int total = ssm_size + pbx_size + ang_size;
    char* state = malloc(total);
    if (state) {
        memcpy(state,                       s->ssm_state, ssm_size);
        memcpy(state + ssm_size,            s->prev_Bx,   pbx_size);
        memcpy(state + ssm_size + pbx_size, s->cum_angle, ang_size);
        *state_size = total;
    }
    return state;
}

void set_internal_state(Mamba* mamba, char* state, int state_size) {
    Config* p = &mamba->config;
    RunState* s = &mamba->state;
    int d_state_half = p->d_state / 2;
    unsigned int ssm_size = p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float);
    unsigned int pbx_size = p->n_layers * p->nheads * p->headdim * p->d_state * sizeof(float);
    unsigned int ang_size = p->n_layers * p->nheads * d_state_half * sizeof(float);
    if ((unsigned int)state_size == ssm_size + pbx_size + ang_size) {
        memcpy(s->ssm_state, state,                       ssm_size);
        memcpy(s->prev_Bx,   state + ssm_size,            pbx_size);
        memcpy(s->cum_angle, state + ssm_size + pbx_size, ang_size);
    }
}

void free_run_state(RunState* s) {
    free(s->input);
    free(s->hidden_state);
    free(s->proj);
    free(s->x);
    free(s->B_heads);
    free(s->C_heads);
    free(s->dt);
    free(s->Bx);
    free(s->y);
    free(s->mlp_gate);
    free(s->mlp_up);
    free(s->logits);
    free(s->ssm_state);
    free(s->prev_Bx);
    free(s->cum_angle);
}

void memory_map_weights(MambaWeights* w, Config* p, float* ptr) {
    unsigned long long n_layers = p->n_layers;
    int d_state_half = p->d_state / 2;
    int d_in_proj = 2 * p->d_inner + 2 * p->d_state + 2 * p->nheads + d_state_half;

    w->token_embedding_table = ptr; ptr += p->rounded_vocab_size * p->dim;
    w->mixer_norm  = ptr;           ptr += n_layers * p->dim;
    w->in_proj     = ptr;           ptr += n_layers * d_in_proj * p->dim;
    w->A           = ptr;           ptr += n_layers * p->nheads;
    w->D           = ptr;           ptr += n_layers * p->nheads;
    w->dt_bias     = ptr;           ptr += n_layers * p->nheads;
    w->B_norm      = ptr;           ptr += n_layers * p->d_state;
    w->C_norm      = ptr;           ptr += n_layers * p->d_state;
    w->B_bias      = ptr;           ptr += n_layers * p->nheads * p->d_state;
    w->C_bias      = ptr;           ptr += n_layers * p->nheads * p->d_state;
    w->out_proj    = ptr;           ptr += n_layers * p->dim * p->d_inner;
    w->mlp_norm    = ptr;           ptr += n_layers * p->dim;
    w->w_gate      = ptr;           ptr += n_layers * p->d_mlp_inner * p->dim;
    w->w_up        = ptr;           ptr += n_layers * p->d_mlp_inner * p->dim;
    w->w_down      = ptr;           ptr += n_layers * p->dim * p->d_mlp_inner;
    w->final_norm  = ptr;           ptr += p->dim;
    w->lm_head = p->shared_classifier ? w->token_embedding_table : ptr;
}

void load_model_file(char* model_path, Config* config, MambaWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", model_path); exit(EXIT_FAILURE); }
    // magic number 'Mmb3' = 0x4d6d6233
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != 0x4d6d6233) {
        fprintf(stderr, "Invalid magic: %x (expected Mamba3: 0x4d6d6233)\n", magic);
        exit(EXIT_FAILURE);
    }
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 3) {
        fprintf(stderr, "Invalid version: %d (expected 3 for Mamba3)\n", version);
        exit(EXIT_FAILURE);
    }
    // config: n_layers, vocab_size, dim, d_inner, d_state, headdim, d_mlp_inner, shared_classifier
    if (fread(&config->n_layers,         sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->vocab_size,       sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->dim,              sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_inner,          sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_state,          sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->headdim,          sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_mlp_inner,      sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->shared_classifier,sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    // derived values
    config->nheads = config->d_inner / config->headdim;
    if (config->vocab_size % 16 != 0) {
        config->rounded_vocab_size = config->vocab_size + (16 - config->vocab_size % 16);
    } else {
        config->rounded_vocab_size = config->vocab_size;
    }
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(model_path, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + (256 / 4);
    memory_map_weights(weights, config, weights_ptr);
}

void load_model(Mamba* m, char* model_path) {
    // read the Config and the Weights from the model file
    load_model_file(model_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, &m->config);
}

void free_model(Mamba* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

float softplus(float x) {
    return logf(1.0f + expf(x));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * weight[j] * ss;
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int d, int n) {
    // w[d,n] @ x[n] -> xout[d]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Apply data-dependent RoPE to x in-place.
// x:      (nheads, d_state) — B or C projection to rotate
// angles: (nheads, d_state/2) — cumulative rotation angles (already negated)
// Rotates each (x[2j], x[2j+1]) pair by angles[j] per head.
void apply_rope(float* x, float* angles, int nheads, int d_state) {
    int half = d_state / 2;
    #pragma omp parallel for
    for (int h = 0; h < nheads; h++) {
        float* xh = x + h * d_state;
        float* ah = angles + h * half;
        for (int j = 0; j < half; j++) {
            float x1 = xh[2 * j];
            float x2 = xh[2 * j + 1];
            float c = cosf(ah[j]);
            float s = sinf(ah[j]);
            xh[2 * j]     = c * x1 - s * x2;
            xh[2 * j + 1] = s * x1 + c * x2;
        }
    }
}

// Forward one Mamba-3 layer: SSM mixer (with trapezoidal recurrence + RoPE) + SwiGLU MLP.
// input is modified in-place (residual stream).
void forward_layer(Mamba* mamba, unsigned long long l, float* input) {
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim, d_inner = p->d_inner, d_state = p->d_state;
    int headdim = p->headdim, nheads = p->nheads;
    int d_state_half = d_state / 2;
    int d_in_proj = 2 * d_inner + 2 * d_state + 2 * nheads + d_state_half;

    // Persistent per-layer state
    float* ssm_state = s->ssm_state + l * nheads * headdim * d_state;
    float* prev_Bx   = s->prev_Bx   + l * nheads * headdim * d_state;
    float* cum_angle = s->cum_angle + l * nheads * d_state_half;

    // Layer weight pointers
    float* mixer_norm_w = w->mixer_norm + l * dim;
    float* in_proj_w    = w->in_proj    + l * d_in_proj * dim;
    float* A_w          = w->A          + l * nheads;
    float* D_w          = w->D          + l * nheads;
    float* dt_bias_w    = w->dt_bias    + l * nheads;
    float* B_norm_w     = w->B_norm     + l * d_state;
    float* C_norm_w     = w->C_norm     + l * d_state;
    float* B_bias_w     = w->B_bias     + l * nheads * d_state;
    float* C_bias_w     = w->C_bias     + l * nheads * d_state;
    float* out_proj_w   = w->out_proj   + l * dim * d_inner;
    float* mlp_norm_w   = w->mlp_norm   + l * dim;
    float* w_gate_w     = w->w_gate     + l * p->d_mlp_inner * dim;
    float* w_up_w       = w->w_up       + l * p->d_mlp_inner * dim;
    float* w_down_w     = w->w_down     + l * dim * p->d_mlp_inner;

    // ====================================================================
    // SSM Mixer
    // ====================================================================

    // Pre-normalization (Llama-style)
    rmsnorm(s->hidden_state, input, mixer_norm_w, dim);

    // Input projection → (z, x, B, C, dt, lam, theta)
    matmul(s->proj, s->hidden_state, in_proj_w, d_in_proj, dim);

    // Split projection buffer
    float* z     = s->proj;
    float* x_raw = s->proj + d_inner;
    float* B_raw = s->proj + 2 * d_inner;
    float* C_raw = s->proj + 2 * d_inner + d_state;
    float* dt_raw= s->proj + 2 * d_inner + 2 * d_state;
    float* lam   = s->proj + 2 * d_inner + 2 * d_state + nheads;
    float* theta = s->proj + 2 * d_inner + 2 * d_state + 2 * nheads;

    // Copy x into scratch buffer (nheads, headdim)
    memcpy(s->x, x_raw, d_inner * sizeof(float));

    // Discretization: dt = softplus(dt + dt_bias), lam = sigmoid(lam)
    float* dt = s->dt;
    for (int h = 0; h < nheads; h++) {
        dt[h]  = softplus(dt_raw[h] + dt_bias_w[h]);
        lam[h] = sigmoid(lam[h]);
    }

    // QK-Normalization on B and C (Section 3.4)
    rmsnorm(B_raw, B_raw, B_norm_w, d_state);
    rmsnorm(C_raw, C_raw, C_norm_w, d_state);

    // Update cumulative RoPE angles: cum_angle[h,j] -= dt[h] * theta[j]
    // (negative because we accumulate −Σ Δ_i*θ_i)
    #pragma omp parallel for
    for (int h = 0; h < nheads; h++) {
        for (int j = 0; j < d_state_half; j++) {
            cum_angle[h * d_state_half + j] -= dt[h] * theta[j];
        }
    }

    // Broadcast B and C to all heads, add head-specific bias, then apply RoPE.
    // QK-norm → add bias → RoPE.
    #pragma omp parallel for
    for (int h = 0; h < nheads; h++) {
        for (int n = 0; n < d_state; n++) {
            s->B_heads[h * d_state + n] = B_raw[n] + B_bias_w[h * d_state + n];
            s->C_heads[h * d_state + n] = C_raw[n] + C_bias_w[h * d_state + n];
        }
    }
    apply_rope(s->B_heads, cum_angle, nheads, d_state);
    apply_rope(s->C_heads, cum_angle, nheads, d_state);

    // Trapezoidal state update + output computation (Proposition 1, Eq. 4):
    //   α   = exp(Δ * A)              — decay (A is already negative: A = -exp(A_log))
    //   β   = (1 − λ) * Δ * α        — left-endpoint coefficient (previous input)
    //   γ   = λ * Δ                  — right-endpoint coefficient (current input)
    //   Bx  = outer(B̄_t, x_t)        — current contribution
    //   h_t = α * h_{t-1} + β * Bx_{t-1} + γ * Bx_t
    //   y_t = h_t^T C̄_t + D * x_t
    #pragma omp parallel for
    for (int h = 0; h < nheads; h++) {
        float alpha     = expf(dt[h] * A_w[h]);
        float beta      = (1.0f - lam[h]) * dt[h] * alpha;
        float gamma_val = lam[h] * dt[h];
        float D_h       = D_w[h];

        for (int pp = 0; pp < headdim; pp++) {
            float xhp = s->x[h * headdim + pp];
            float y_hp = 0.0f;

            for (int n = 0; n < d_state; n++) {
                int idx = h * headdim * d_state + pp * d_state + n;
                float bx = s->B_heads[h * d_state + n] * xhp;
                s->Bx[idx] = bx;
                ssm_state[idx] = alpha    * ssm_state[idx]
                               + beta     * prev_Bx[idx]
                               + gamma_val * bx;
                y_hp += ssm_state[idx] * s->C_heads[h * d_state + n];
            }
            s->y[h * headdim + pp] = y_hp + D_h * xhp;
        }
    }

    // Gate: y = y ⊙ silu(z)
    #pragma omp parallel for
    for (int i = 0; i < d_inner; i++) {
        s->y[i] *= silu(z[i]);
    }

    // Output projection and residual connection
    matmul(s->hidden_state, s->y, out_proj_w, dim, d_inner);
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) { input[i] += s->hidden_state[i]; }

    // Save Bx → prev_Bx for next token's β term
    memcpy(prev_Bx, s->Bx, nheads * headdim * d_state * sizeof(float));

    // ====================================================================
    // SwiGLU MLP  (Llama-style: SwiGLU(x) = W_down(silu(W_gate(x)) ⊙ W_up(x)))
    // ====================================================================

    // Pre-normalization
    rmsnorm(s->hidden_state, input, mlp_norm_w, dim);

    // Gate and Up projections
    matmul(s->mlp_gate, s->hidden_state, w_gate_w, p->d_mlp_inner, dim);
    matmul(s->mlp_up,   s->hidden_state, w_up_w,   p->d_mlp_inner, dim);

    // Fused SwiGLU: gate = silu(gate) * up  (in-place into mlp_gate)
    #pragma omp parallel for
    for (int i = 0; i < p->d_mlp_inner; i++) {
        s->mlp_gate[i] = silu(s->mlp_gate[i]) * s->mlp_up[i];
    }

    // Down projection and residual connection
    matmul(s->hidden_state, s->mlp_gate, w_down_w, dim, p->d_mlp_inner);
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) { input[i] += s->hidden_state[i]; }
}

float* forward(Mamba* mamba, int token) {
    // a few convenience variables
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    float* input = s->input;

    // copy the token embedding into x
    memcpy(input, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        forward_layer(mamba, l, input);
    }

    // final rmsnorm
    rmsnorm(s->hidden_state, input, w->final_norm, p->dim);
    // classifier into logits
    matmul(s->logits, s->hidden_state, w->lm_head, p->rounded_vocab_size, p->dim);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

#define BOS 0
#define EOS 0

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int model_vocab_size) {
    // initialize the byte_pieces array
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    // read header magic
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    if (magic != 0x4d62546b) { fprintf(stderr, "invalid magic number: %x\n", magic); exit(EXIT_FAILURE); }
    // read version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "invalid version: %d\n", version); exit(EXIT_FAILURE); }
    // read vocab_size
    int vocab_size;
    if (fread(&vocab_size, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    // read max_token_length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    // malloc space for the vocab
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    if (!t->vocab) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    t->sorted_vocab = NULL; // initialized lazily
    // read vocab
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // discard initial space if prev_token was EOS
    if (prev_token == EOS && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t add_bos, int8_t add_eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // add_bos != 0 means prepend the BOS token, add_eos != 0 means append the EOS token
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS token, if desired
    if (add_bos) tokens[(*n_tokens)++] = BOS;

    // add_dummy_prefix is not used in Mamba, but it's here for reference
    // prepend a dummy prefix token to the input string, but only if text != ""
    int add_dummy_prefix = 0;
    if (add_dummy_prefix && text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration
    while (1) {
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1) {
                // this merge pair exists in vocab! record its position
                best_id = id;
                best_idx = i;
                break;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS token, if desired
    if (add_eos) tokens[(*n_tokens)++] = EOS;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Mamba *mamba, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', BOS, EOS
    encode(tokenizer, prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // print the first token in the prompt
    if (num_prompt_tokens > 1) {
        char* piece = decode(tokenizer, EOS, prompt_tokens[0]);
        safe_printf(piece);
        fflush(stdout);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the model to get logits for the next token
        float* logits = forward(mamba, token);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the EOS token delimits sequences
        if (next == EOS) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Mamba *mamba, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the model
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the model next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS token ends the Assistant turn
        if (token == EOS) { user_turn = 1; }

        // forward the model to get logits for the next token
        float* logits = forward(mamba, token);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != EOS) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == EOS) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path = NULL;    // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { model_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // load the model using the model.bin file
    Mamba mamba;
    load_model(&mamba, model_path);

    // print the config
    fprintf(stderr, "Mamba-3: vocab=%d (%d), layers=%d, dim=%d, d_inner=%d, "
            "d_state=%d, headdim=%d, nheads=%d, d_mlp_inner=%d\n",
            mamba.config.vocab_size, mamba.config.rounded_vocab_size,
            mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner,
            mamba.config.d_state, mamba.config.headdim, mamba.config.nheads,
            mamba.config.d_mlp_inner);

    if (steps == 0) steps = 256; // override to default len if 0

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, mamba.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, mamba.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&mamba, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&mamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
    return 0;
}
#endif
