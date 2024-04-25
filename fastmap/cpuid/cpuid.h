#if defined(_MSC_VER)
#include <intrin.h>
#define CPUID(info, x)  __cpuidex((int *)(info), x, 0)
#else
#include <cpuid.h>
#define CPUID(info, x)  __cpuid_count(x, 0, (info)[0], (info)[1], (info)[2], (info)[3])
#endif

enum simd_t {
    SIMD_NONE     = 0,        ///< None
    SIMD_SSE      = 1 << 0,   ///< SSE
    SIMD_SSE2     = 1 << 1,   ///< SSE 2
    SIMD_SSE3     = 1 << 2,   ///< SSE 3
    SIMD_SSSE3    = 1 << 3,   ///< SSSE 3
    SIMD_SSE41    = 1 << 4,   ///< SSE 4.1
    SIMD_SSE42    = 1 << 5,   ///< SSE 4.2
    SIMD_FMA3     = 1 << 6,   ///< FMA 3
    SIMD_FMA4     = 1 << 7,   ///< FMA 4
    SIMD_AVX      = 1 << 8,   ///< AVX
    SIMD_AVX2     = 1 << 9,   ///< AVX 2
    SIMD_AVX512   = 1 << 10,  ///< AVX 512
    SIMD_SVML     = 1 << 11,  ///< SVML
};

typedef struct {
    int simd_flags_;
} SIMDFlags;

void SIMDFlags_init(SIMDFlags *flags) {
    unsigned int cpuInfo[4];
    CPUID(cpuInfo, 0x00000001);
    flags->simd_flags_ |= cpuInfo[3] & (1 << 25) ? SIMD_SSE   : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[3] & (1 << 26) ? SIMD_SSE2  : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 0)  ? SIMD_SSE3  : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 9)  ? SIMD_SSSE3 : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 19) ? SIMD_SSE41 : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 20) ? SIMD_SSE42 : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 12) ? SIMD_FMA3  : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[2] & (1 << 28) ? SIMD_AVX   : SIMD_NONE;

    CPUID(cpuInfo, 0x00000007);
    flags->simd_flags_ |= cpuInfo[1] & (1 << 5)  ? SIMD_AVX2  : SIMD_NONE;
    flags->simd_flags_ |= cpuInfo[1] & (1 << 16) ? SIMD_AVX512: SIMD_NONE;

    CPUID(cpuInfo, 0x80000001);
    flags->simd_flags_ |= cpuInfo[2] & (1 << 16) ? SIMD_FMA4  : SIMD_NONE;

    CPUID(cpuInfo, 0x8000000A);
    flags->simd_flags_ |= cpuInfo[1] & (1 << 2)  ? SIMD_SVML  : SIMD_NONE;
}

int SIMDFlags_hasSSE(const SIMDFlags *flags)   { return flags->simd_flags_ & SIMD_SSE;   }
int SIMDFlags_hasSSE2(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_SSE2;  }
int SIMDFlags_hasSSE3(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_SSE3;  }
int SIMDFlags_hasSSSE3(const SIMDFlags *flags) { return flags->simd_flags_ & SIMD_SSSE3; }
int SIMDFlags_hasSSE41(const SIMDFlags *flags) { return flags->simd_flags_ & SIMD_SSE41; }
int SIMDFlags_hasSSE42(const SIMDFlags *flags) { return flags->simd_flags_ & SIMD_SSE42; }
int SIMDFlags_hasFMA3(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_FMA3;  }
int SIMDFlags_hasFMA4(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_FMA4;  }
int SIMDFlags_hasAVX(const SIMDFlags *flags)   { return flags->simd_flags_ & SIMD_AVX;   }
int SIMDFlags_hasAVX2(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_AVX2;  }
int SIMDFlags_hasAVX512(const SIMDFlags *flags){ return flags->simd_flags_ & SIMD_AVX512;}
int SIMDFlags_hasSVML(const SIMDFlags *flags)  { return flags->simd_flags_ & SIMD_SVML;  }
