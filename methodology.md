# Detailed Methodology

This document provides the complete mathematical formulation for the feature extraction pipeline used in our study: _"Exploiting Temporal Amplitude Envelopes for Robust Deepfake Audio Detection"_.

## 1. Overview

Utilizing a supervised learning approach trained on labeled authentic and spoofed recordings, the objective of this study is to detect artificial audio manipulations by leveraging advanced temporal amplitude envelope analysis, modulation spectrum features, background/foreground spectral separation, and refined loudness analysis. The overall pipeline consists of dataset construction, baseline and enhanced feature extraction, multi-model training, and performance evaluation.

---

## 2. Baseline Feature Extraction

The baseline system extracts 40 MFCC coefficients per frame and aggregates them over time by taking the temporal mean. If $\text{MFCC}_t \in \mathbb{R}^{40}$ denotes the MFCC vector at frame $t$ and $T$ is the number of frames in the utterance, the global MFCC mean is computed as:

$$
\overline{\text{MFCC}} = \frac{1}{T} \sum_{t=1}^{T} \text{MFCC}_{t}
$$

The resulting 40-dimensional vector serves as the baseline feature representation.

### Enhanced MFCC Features with Delta Coefficients

In addition to the baseline MFCC means, temporal dynamics are captured through first-order (delta) and second-order (delta-delta) derivatives. For each MFCC coefficient, the delta at frame $t$ is computed as:

$$
\Delta \text{MFCC}_{t} = \frac{\sum_{\theta=1}^{\Theta} \theta \big( \text{MFCC}_{t+\theta} - \text{MFCC}_{t-\theta} \big)}{2 \sum_{\theta=1}^{\Theta} \theta^{2}}
$$

where $\Theta$ is the delta window size. Second-order derivatives are obtained by applying the same operation to the delta sequence. These derived parameters improve sensitivity to unnatural temporal transitions in synthetic speech.

---

## 3. Enhanced Feature Extraction

The advanced feature extraction pipeline extends the baseline features by introducing four additional categories of descriptors that characterize temporal, spectral, modulation, and background/foreground aspects.

### A. Amplitude Envelope Feature Extraction

Let $x[n]$ denote the discrete-time speech signal. The amplitude envelope $A[n]$ is approximated by the absolute value:

$$
A[n] = |x[n]|
$$

From $A[n]$, standard statistical descriptors are computed:

$$
\mu_A = \frac{1}{N} \sum_{n=1}^{N} A[n]
$$

$$
\sigma_A = \sqrt{\frac{1}{N} \sum_{n=1}^{N} \big(A[n] - \mu_A\big)^{2}}
$$

$$
R_A = \max_{n} A[n] - \min_{n} A[n]
$$

where $N$ is the number of samples in the utterance. These features describe the central tendency, variability, and dynamic range of the amplitude envelope.

#### Temporal Segmentation & Jump Detection

To detect unnatural amplitude patterns, the envelope is partitioned into $K = 10$ non-overlapping segments of equal length $L = N/K$. For segment index $i$, the segment mean is:

$$
\mu_i = \frac{1}{L} \sum_{n=iL}^{(i+1)L - 1} A[n]
$$

Temporal discontinuities are quantified via intersegment amplitude jumps:

$$
\Delta_i = \mu_{i+1} - \mu_i, \qquad i = 0, 1, \ldots, K-2
$$

The maximum jump magnitude and jump variance are computed as:

$$
\Delta_{\max} = \max_{i} \big|\Delta_i\big|
$$

$$
\sigma_{\Delta}^{2} = \frac{1}{K - 1} \sum_{i=0}^{K-2} \big(\Delta_i - \bar{\Delta}\big)^{2}
$$

where $\bar{\Delta}$ is the mean of the amplitude jumps. These features capture sudden level changes that are characteristic of splicing or stitching artifacts in fake audio.

#### Rise Pattern Analysis

To focus on upward and downward changes separately, the set of positive and negative jumps is defined as $\Delta^{+}$ and $\Delta^{-}$. The sudden rise ratio is given by:

$$
\rho_{\text{rise}} = \frac{|\Delta^{+}|}{K - 1}
$$

#### Multi-scale Temporal Analysis

To capture multi-scale temporal trends, the envelope is smoothed using sliding rectangular windows of different lengths:

- $W_s$ (Short): e.g., 50ms
- $W_l$ (Long): e.g., 200ms

$$
A_{\text{short}}[n] = \frac{1}{W_s} \sum_{k=0}^{W_s - 1} A[n - k]
$$

$$
A_{\text{long}}[n] = \frac{1}{W_l} \sum_{k=0}^{W_l - 1} A[n - k]
$$

The corresponding average slopes are computed as:

$$
S_{\text{short}} = \frac{1}{M - 1} \sum_{m=1}^{M-1} \big(A_{\text{short}}[m+1] - A_{\text{short}}[m]\big)
$$

$$
S_{\text{long}} = \frac{1}{P - 1} \sum_{p=1}^{P-1} \big(A_{\text{long}}[p+1] - A_{\text{long}}[p]\big)
$$

where $M$ and $P$ denote the lengths of the short and long smoothed sequences, respectively. These slopes summarize gradual rising or falling trends at different temporal scales.

#### Statistical Distribution Features

Higher-order moments of the envelope distribution (Skewness $\gamma_A$ and Kurtosis $\kappa_A$) are also computed:

$$
\gamma_A = \frac{E\big[(A[n] - \mu_A)^{3}\big]}{\sigma_A^{3}}, \qquad \kappa_A = \frac{E\big[(A[n] - \mu_A)^{4}\big]}{\sigma_A^{4}} - 3
$$

These capture asymmetry and tailedness, which can be indicative of unnatural dynamic behavior.

---

### B. Loudness Analysis with Jump Detection

Short-time loudness is estimated using frame-wise RMS energy. For frame index $m$ and hop size $H$ in samples:

$$
L[m] = \sqrt{\frac{1}{H} \sum_{n = mH}^{(m+1)H - 1} x^{2}[n]}
$$

Loudness differences between successive frames are computed as:

$$
\Delta L[m] = L[m+1] - L[m]
$$

Let $\sigma_{\Delta L}$ denote the standard deviation of the sequence $\Delta L[m]$. The number of loudness spikes exceeding a three standard deviation threshold is then defined as:

$$
f(\Delta L[m]) = \begin{cases} \Delta L[m], & \text{if } \Delta L[m] > 3\sigma_{\Delta L}, \\ 0, & \text{otherwise}, \end{cases}
$$

Then the loudness–spike measure is defined as:

$$
N_{\text{spike}} = \sum_{m} f(\Delta L[m])
$$

This separates the definition of frame-level loudness $L[m]$ from the spike count $N_{\text{spike}}$.

---

### C. Enhanced Modulation Spectrum Features

Modulation domain features are derived from windowed frames of the amplitude envelope. Let $w[n]$ be a Hanning window:

$$
w[n] = 0.5 \left( 1 - \cos \frac{2 \pi n}{F - 1} \right)
$$

The $i$-th windowed frame is:

$$
f_i[n] = A[iH + n] \, w[n]
$$

and its modulation power spectrum is given by:

$$
M_i(\omega) = \big| \text{FFT}\big(f_i[n]\big) \big|^{2}
$$

Modulation power is summarized over three bands (0-20 Hz, 20-50 Hz, and 50-100 Hz) using statistics such as mean, standard deviation, and range.

---

### D. Background/Foreground Analysis

Time-frequency structure is analyzed using the magnitude of the Short Time Fourier Transform (STFT), $X(k,m)$:

$$
X(k,m) = \left| \sum_{n=0}^{F-1} x[n + mH] \, w[n] \, e^{-j 2 \pi k n / F} \right|
$$

A background estimate for each frequency bin is defined as the 10th percentile of its magnitude over time:

$$
B_k = \mathrm{percentile}_{10}\big( X(k, m) \big)
$$

A foreground mask separates high energy regions ($X > 2 B_k$):

$$
M_{\text{fg}}(k, m) = \begin{cases} 1, & \text{if } X(k,m) > 2 B_k \\ 0, & \text{otherwise} \end{cases}
$$

The background-to-foreground energy ratio is computed as:

$$
\rho_{\text{bg/fg}} = \frac{E\big[ X(k,m) \mid M_{\text{fg}}(k,m) = 0 \big]}{E\big[ X(k,m) \mid M_{\text{fg}}(k,m) = 1 \big]}
$$

#### Background Temporal Stability

To characterize temporal stability of the background, a background-only temporal energy trajectory is defined as:

$$
B_{\text{temp}}[m] = \frac{1}{K} \sum_{k=1}^{K} X(k,m) \, \big( 1 - M_{\text{fg}}(k,m) \big)
$$

Sudden background changes are counted whenever the frame-to-frame change exceeds a fixed threshold ($2\sigma$):

$$
N_{\text{bg}} = \sum_{m=1}^{M-1} \mathbf{1}_{\big\{ \big| B_{\text{temp}}[m+1] - B_{\text{temp}}[m] \big| > 2 \sigma_{B_{\text{temp}}} \big\}}
$$

where $\mathbf{1}$ is the indicator function. The threshold of $2\sigma$ targets fluctuations outside the typical 95% confidence interval of background variation.

---

## 4. Feature Vector Construction

All features are concatenated into a single utterance-level feature vector $\mathbf{F}$:

$$
\mathbf{F} = \big\{ \text{MFCC}, \, \Delta \text{MFCC}, \, \Delta\Delta \text{MFCC}, \, F_{\text{amp}}, \, F_{\text{loud}}, \, F_{\text{mod}}, \, F_{\text{spec}}, \, F_{\text{HPSS}}, \, F_{\text{bg/fg}} \big\}
$$

Where:

- $F_{\text{amp}}$: Amplitude envelope statistics & jumps
- $F_{\text{loud}}$: Loudness dynamics
- $F_{\text{mod}}$: Modulation spectrum statistics
- $F_{\text{bg/fg}}$: Background/Foreground stability metrics
