/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __BxDF_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __BxDF_HLSLI__

#include "../../Config.h"    

#include "../../Utils/Math/MathConstants.hlsli"

#include "BxDFConfig.hlsli"

#include "../../Scene/ShadingData.hlsli"
#include "../../Utils/Math/MathHelpers.hlsli"
#include "../../Utils/Color/ColorHelpers.hlsli"
#include "Fresnel.hlsli"
#include "Microfacet.hlsli"

#include "../../StablePlanes.hlsli"

// Minimum cos(theta) for the incident and outgoing vectors.
// Some BSDF functions are not robust for cos(theta) == 0.0,
// so using a small epsilon for consistency.
static const float kMinCosTheta = 1e-6f;

// Because sample values must be strictly less than 1, it’s useful to define a constant, OneMinusEpsilon, that represents the largest 
// representable floating-point constant that is less than 1. (https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Sampling_Interface)
static const float OneMinusEpsilon = 0x1.fffffep-1;

#define GGXSamplingNDF          0
#define GGXSamplingVNDF         1
#define GGXSamplingBVNDF        2


// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF         1

#define GGXSampling             GGXSamplingBVNDF

// When deciding a lobe to sample, expand and reuse the random sample - losing at precision but gaining on performance when using costly LD sampler
#define RecycleSelectSamples    1

// We clamp the GGX width parameter to avoid numerical instability.
// In some computations, we can avoid clamps etc. if 1.0 - alpha^2 != 1.0, so the epsilon should be 1.72666361e-4 or larger in fp32.
// The the value below is sufficient to avoid visible artifacts.
// Falcor used to clamp roughness to 0.08 before the clamp was removed for allowing delta events. We continue to use the same threshold.
static const float kMinGGXAlpha = 0.0064f;

// Note: preGeneratedSample argument value in 'sample' interface is a vector of 3 or 4 [0, 1) random numbers, generated with the SampleGenerator and 
// depending on configuration will either be a pseudo-random or quasi-random.
// Some quasi-random (Low Discrepancy / Stratified) samples are such that dimensions are designed to work well in pairs, so ideally use .xy for lobe
// projection sample and .z for lobe selection (if used).
// For more info see https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Stratified_Sampling

/** Lambertian diffuse reflection.
    f_r(wi, wo) = albedo / pi
*/
struct DiffuseReflectionLambert // : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0,0,0);

        return M_1_PI * albedo * wo.z;
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        wo = sample_cosine_hemisphere_concentric(preGeneratedSample.xy, pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = float3(0,0,0);
            lobeP = 0.0;
            return false;
        }

        weight = albedo;
        lobeP = 1.0;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }
};



/* 
 //faceting on CC head, https://app.asana.com/0/1208704467141427/1208971573306148
#define SSS_SAMPLING_DISK_AXIS_0_WEIGHT 0.5
#define SSS_SAMPLING_DISK_AXIS_1_WEIGHT 0.25
#define SSS_SAMPLING_DISK_AXIS_2_WEIGHT 0.25
/*/
#define SSS_SAMPLING_DISK_AXIS_0_WEIGHT 1
#define SSS_SAMPLING_DISK_AXIS_1_WEIGHT 0
#define SSS_SAMPLING_DISK_AXIS_2_WEIGHT 0
//*/

#define SSS_SAMPLING_DISK_CHANNEL_0_WEIGHT 1.0 / 3.0
#define SSS_SAMPLING_DISK_CHANNEL_1_WEIGHT 1.0 / 3.0
#define SSS_SAMPLING_DISK_CHANNEL_2_WEIGHT 1.0 / 3.0

#define INVALID_UINT_VALUE 0xFFFFFFFFu

// Structures
struct SSSInfo
{
    float3 position;
    uint geometryInstanceID;
    float3 scatterDistance;
    uint intersection;
    static SSSInfo make( float3 p, uint geometryInstanceID, float3 dist, uint intersect)
    {
        SSSInfo ret;
        ret.position = p;
        ret.geometryInstanceID = geometryInstanceID;
        ret.scatterDistance = dist;
        ret.intersection = intersect;
        return ret;
    }
};

struct SSSSample
{
    uint geometryInstanceID;
    int triangleId;
    float2 barycentrics;
    float3 position; // in world
    float3 geometricNormal;
    float3 normal;
    uint intersection;
    static SSSSample makeZero()
    {
        SSSSample ret;
        ret.geometryInstanceID = 0;
        ret.triangleId = 0;
        ret.barycentrics = float2(0, 0);
        ret.position = float3(0, 0, 0);
        ret.geometricNormal = float3(0, 0, 0);
        ret.normal = float3(0, 0, 0);
        ret.intersection = INVALID_UINT_VALUE;
        return ret;
    }
    static SSSSample make( float2 bary, float3 pos, float3 normal, float3 geometricNormal, uint geometryInstanceID, uint primitiveID, uint intersection )
    {
        SSSSample ret;
        ret.geometryInstanceID = geometryInstanceID;
        ret.triangleId = primitiveID;
        ret.barycentrics = bary;
        ret.position = pos;
        ret.geometricNormal = geometricNormal;
        ret.normal = normal;
        ret.intersection = intersection;
        return ret;
    }
};

struct BSDFFrame
{
    float3 n;
    float3 t;
    float3 b;
};

float3 sss_diffusion_profile_pdf_vectorized(in const float radius, in const float3 scatterDistance) {
    if (radius <= 0) {
        return (0.25 / M_PI) / max(float3(0.000001,0.000001,0.000001), scatterDistance);
    }
    const float3 rd = radius / scatterDistance;
    return (exp(-rd) + exp(-rd / 3.0)) / max(float3(0.000001,0.000001,0.000001), (8.0 * M_PI * scatterDistance * radius));// divide by r to convert from polar to cartesian
}

uint sss_sampling_axis_index(in const float xiAxis) {
    if (xiAxis < SSS_SAMPLING_DISK_AXIS_0_WEIGHT) {
        return 0;
    } else if (xiAxis < (SSS_SAMPLING_DISK_AXIS_0_WEIGHT + SSS_SAMPLING_DISK_AXIS_1_WEIGHT)) {
        return 1;
    } else {
        return 2;
    }
}
void sss_sampling_axis(in const uint axis, in const BSDFFrame frame, out BSDFFrame projectionFrame) {
    if (axis == 0) {
        projectionFrame.t = frame.t;
        projectionFrame.b = frame.b;
        projectionFrame.n = frame.n;
    } else if (axis == 1) {
        projectionFrame.t = frame.b;
        projectionFrame.b = frame.n;
        projectionFrame.n = frame.t;
    } else {
        projectionFrame.t = frame.n;
        projectionFrame.b = frame.t;
        projectionFrame.n = frame.b;
    }
}
// https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/s2013_bssrdf_slides.pdf
// https://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Subsurface_Reflection_Functions#SeparableBSSRDF::Pdf_Sp

float sss_sampling_disk_pdf(
    in float3 sssSampleDistance,
    //in float3 position,
    in BSDFFrame frame,
    //in float3 sssSamplePosition,
    in float3 sampleNormal,
    in float3 scatterDistance)
{
    float3 d = sssSampleDistance; //samplePosition - position;
    
    // Transform vector into local space using frame
    float3 dLocal = float3(
        dot(frame.n, d),
        dot(frame.t, d),
        dot(frame.b, d)
    );
    
    // Compute projected distances
    float3 rProj = float3(
        sqrt(dLocal.y * dLocal.y + dLocal.z * dLocal.z),
        sqrt(dLocal.z * dLocal.z + dLocal.x * dLocal.x),
        sqrt(dLocal.x * dLocal.x + dLocal.y * dLocal.y)
    );
    
    // Compute local normal components using absolute values
    float3 nLocal = abs(float3(
        dot(frame.n, sampleNormal),
        dot(frame.t, sampleNormal),
        dot(frame.b, sampleNormal)
    ));
    
    // Sampling weights for axes and channels
    static const float3 axisProb = { SSS_SAMPLING_DISK_AXIS_0_WEIGHT,
                                   SSS_SAMPLING_DISK_AXIS_1_WEIGHT,
                                   SSS_SAMPLING_DISK_AXIS_2_WEIGHT };
    
    static const float3 channelProb = { SSS_SAMPLING_DISK_CHANNEL_0_WEIGHT,
                                      SSS_SAMPLING_DISK_CHANNEL_1_WEIGHT,
                                      SSS_SAMPLING_DISK_CHANNEL_2_WEIGHT };
    
    float pdf = 0.0f;
    
    // Calculate PDF contributions for each axis and channel combination
    // Using vectorized version for better performance
    
    // Axis 0
    float3 pdfAxis = sss_diffusion_profile_pdf_vectorized(rProj[0], scatterDistance) * 
                     axisProb[0] * channelProb[0] * nLocal[0];
    pdf += pdfAxis[0] + pdfAxis[1] + pdfAxis[2];
    
    // Axis 1
    pdfAxis = sss_diffusion_profile_pdf_vectorized(rProj[1], scatterDistance) * 
              axisProb[1] * channelProb[1] * nLocal[1];
    pdf += pdfAxis[0] + pdfAxis[1] + pdfAxis[2];
    
    // Axis 2
    pdfAxis = sss_diffusion_profile_pdf_vectorized(rProj[2], scatterDistance) * 
              axisProb[2] * channelProb[2] * nLocal[2];
    pdf += pdfAxis[0] + pdfAxis[1] + pdfAxis[2];
    
    return pdf;
}

#define LOG2_E 1.44269504089
//// https://zero-radiance.github.io/post/sampling-diffusion/
//// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
//// 'u' is the random number (the value of the CDF): [0, 1).
//// rcp(s) = 1 / ShapeParam = ScatteringDistance.
//// 'r' is the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
//// rcp(Pdf) is the reciprocal of the corresponding PDF value.
float sampleBurleyDiffusionProfileAnalytical(in float u, in const float rcpS) {
    u = 1 - u;// Convert CDF to CCDF; the resulting value of (u != 0)

    const float g = 1 + (4 * u) * (2 * u + sqrt(1 + (4 * u) * u));
    const float n = exp2(log2(g) * (-1.0/3.0));// g^(-1/3)
    const float p = (g * n) * n;// g^(+1/3)
    const float c = 1 + p + n;// 1 + g^(+1/3) + g^(-1/3)
    const float x = (3 / LOG2_E) * log2(c / (4 * u));// 3 * Log[c / (4 * u)]

    // x      = s * r
    // exp_13 = Exp[-x/3] = Exp[-1/3 * 3 * Log[c / (4 * u)]]
    // exp_13 = Exp[-Log[c / (4 * u)]] = (4 * u) / c
    // exp_1  = Exp[-x] = exp_13 * exp_13 * exp_13
    // expSum = exp_1 + exp_13 = exp_13 * (1 + exp_13 * exp_13)
    // rcpExp = rcp(expSum) = c^3 / ((4 * u) * (c^2 + 16 * u^2))
    const float rcpExp = ((c * c) * c) / ((4 * u) * ((c * c) + (4 * u) * (4 * u)));

    return x * rcpS; // r
}

float sss_diffusion_profile_sample(in const float xi, in const float scatterDistance) {
    return sampleBurleyDiffusionProfileAnalytical(xi, scatterDistance);
}

float3 sss_diffusion_profile_scatterDistance(in const float3 surfaceAlbedo) {
    const float3 a = surfaceAlbedo - (float3)(0.8);
    return (float3)1.9 - surfaceAlbedo + 3.5 * a * a;
}

float GetPerpendicularScalingFactor(float SurfaceAlbedo)
{
    // add abs() to match the formula in the original paper
    return 1.85 - SurfaceAlbedo + 7 * pow(abs(SurfaceAlbedo - 0.8), 3);
}

float3 GetPerpendicularScalingFactor(float3 SurfaceAlbedo)
{
    return float3(
        GetPerpendicularScalingFactor(SurfaceAlbedo.r),
        GetPerpendicularScalingFactor(SurfaceAlbedo.g),
        GetPerpendicularScalingFactor(SurfaceAlbedo.b)
    );
}

// Method 2: Ideal diffuse transmission at the surface. More appropriate for rough surface.
// Average relative error: 3.9% (reference to MC)
float GetDiffuseSurfaceScalingFactor(float SurfaceAlbedo)
{
    return 1.9 - SurfaceAlbedo + 3.5 * pow(SurfaceAlbedo - 0.8, 2);
}

float3 GetDiffuseSurfaceScalingFactor(float3 SurfaceAlbedo)
{
    return float3(
        GetDiffuseSurfaceScalingFactor(SurfaceAlbedo.r),
        GetDiffuseSurfaceScalingFactor(SurfaceAlbedo.g),
        GetDiffuseSurfaceScalingFactor(SurfaceAlbedo.b)
    );
}

// Method 3: The spectral of diffuse mean free path on the surface.
// Average relative error: 7.7% (reference to MC)
float GetSearchLightDiffuseScalingFactor(float SurfaceAlbedo)
{
    return 3.5 + 100 * pow(SurfaceAlbedo - 0.33, 4);
}

float3 GetSearchLightDiffuseScalingFactor(float3 SurfaceAlbedo)
{
    return float3(
        GetSearchLightDiffuseScalingFactor(SurfaceAlbedo.r),
        GetSearchLightDiffuseScalingFactor(SurfaceAlbedo.g),
        GetSearchLightDiffuseScalingFactor(SurfaceAlbedo.b)
    );
}

float3 GetMeanFreePathFromDiffuseMeanFreePath(float3 SurfaceAlbedo, float3 DiffuseMeanFreePath)
{
    return DiffuseMeanFreePath * (GetPerpendicularScalingFactor(SurfaceAlbedo) / GetSearchLightDiffuseScalingFactor(SurfaceAlbedo));
}

float3 GetDiffuseMeanFreePathFromMeanFreePath(float3 SurfaceAlbedo, float3 MeanFreePath)
{
    return MeanFreePath * (GetSearchLightDiffuseScalingFactor(SurfaceAlbedo) / GetPerpendicularScalingFactor(SurfaceAlbedo));
}

float3 GetSearchLightDiffuseScalingFactor3D(float3 SurfaceAlbedo)
{
	float3 Value = SurfaceAlbedo - 0.33;
	return 3.5 + 100 * Value * Value * Value * Value;
}

float3 GetPerpendicularScalingFactor3D(float3 SurfaceAlbedo)
{
	float3 Value = abs(SurfaceAlbedo - 0.8);
	return 1.85 - SurfaceAlbedo + 7 * Value * Value * Value;
}

// Mathmatically matching based on diffusion coefficient instead of burley's approximation. However, it leads to incorrect result as 
// we use burley's approximation (IOR=1.0) for screenspace diffuse scattering.
//float3 Alpha = 1 - exp(-11.43 * SurfaceAlbedo + 15.38 * SurfaceAlbedo * SurfaceAlbedo - 13.91 * SurfaceAlbedo * SurfaceAlbedo * SurfaceAlbedo);
//return DMFP * sqrt(3 * (1 - Alpha) / (2 - Alpha));
float3 GetMFPFromDMFPCoeff(float3 DMFPSurfaceAlbedo, float3 MFPSurfaceAlbedo, float Dmfp2MfpMagicNumber = 0.6f)
{
	return Dmfp2MfpMagicNumber * GetPerpendicularScalingFactor3D(MFPSurfaceAlbedo) / GetSearchLightDiffuseScalingFactor3D(DMFPSurfaceAlbedo);
}

float3 GetMFPFromDMFPApprox(float3 SurfaceAlbedo, float3 TargetSurfaceAlbedo, float3 DMFP)
{
	return GetMFPFromDMFPCoeff(SurfaceAlbedo, TargetSurfaceAlbedo) * DMFP;
}

float3 GetDMFPFromMFPApprox(float3 SurfaceAlbedo, float3 MFP)
{
	float3 MFPFromDMFPCoeff = GetMFPFromDMFPCoeff(SurfaceAlbedo, SurfaceAlbedo);
	return MFP / MFPFromDMFPCoeff;
}

#if 0
// With world unit scale 
float4 GetSubsurfaceProfileMFPInCm(int SubsurfaceProfileInt)
{
	float4 DMFP = GetSubsurfaceProfileDMFPInCm(SubsurfaceProfileInt);
	float4 SurfaceAlbedo = GetSubsurfaceProfileSurfaceAlbedo(SubsurfaceProfileInt);

	return float4(GetMFPFromDMFPApprox(SurfaceAlbedo.xyz, SurfaceAlbedo.xyz, DMFP.xyz),0.0f);
}
#endif

float sss_sampling_scatterDistance(in const uint channel, in const float3 scatterDistance) {
    return scatterDistance[channel];
}

    float disney_schlickWeight(in const float a)
    {
        const float b = clamp(1.0 - a, 0.0, 1.0);
        const float bb = b * b;
        return bb * bb * b;
    }

    float disney_diffuseLambertWeight(in const float fv, in const float fl)
    {
        return (1.0 - 0.5 * fl) * (1.0 - 0.5 * fv);
    }

    float disney_diffuseLambertWeightSingle(in const float f)
    {
        return 1.0 - 0.5 * f;
    }
    
    
    float3 sss_diffusion_profile_evaluate(in const float radius, in const float3 scatterDistance)
    {
        if (radius <= 0)
        {
            return (float3)(0.25 / M_PI) / max((float3)0.000001, scatterDistance);
        }
        const float3 rd = radius / scatterDistance;
        return (exp(-rd) + exp(-rd / 3.0)) / max((float3)0.000001, (8.0 * M_PI * scatterDistance * radius));
    }
    
    float3 disney_bssrdf_fresnel_evaluate(in const float3 normal, in const float3 direction)
    {
        const float dotND = dot(normal, direction);
        const float schlick = disney_schlickWeight(dotND);
        const float lambertWeight = disney_diffuseLambertWeightSingle(schlick);
        return (float3)lambertWeight;
    }
    
    void disney_bssrdf_evaluate(in const float3 normal, in const float3 v, in const float distance, in const float3 scatterDistance, in const float3 surfaceAlbedo, out float3 bssrdf)
    {
        const float3 diffusionProfile = surfaceAlbedo * sss_diffusion_profile_evaluate(distance, scatterDistance);

        bssrdf = diffusionProfile / M_PI * disney_bssrdf_fresnel_evaluate(normal, v);
    }

void disney_bssrdf_evaluate(in const float3 normal, 
                            in const float3 v, 
                            in const float3 normalSample, 
                            in const float3 l, 
                            in const float distance, 
                            in const float3 scatterDistance, 
                            in const float3 surfaceAlbedo, 
                            out float3 bssrdf, 
                            out float3 bsdf) {
    const float3 diffusionProfile = surfaceAlbedo * sss_diffusion_profile_evaluate(distance, scatterDistance);

    bssrdf = diffusionProfile / M_PI * disney_bssrdf_fresnel_evaluate(normal, v);
    bsdf = disney_bssrdf_fresnel_evaluate(normalSample, l);
}

struct BssrdfDiffuseReflection
{
    float3 sssMeanFreePath; ///< mean free path
    float3 albedo;  ///< Diffuse albedo.
    //BSDFFrame frame; ///< N, T, B
    float3 pixelNormal;
    float3 sssNormal;
    float3 sssDistance; // sssPosition - position
    float3 scatterDistance;
    float3 DiffuseMeanFreePath;
    float bssrdfPDF;
    float intersectionPDF;
    
    static BssrdfDiffuseReflection make( float3 albedo_,
                                         float3 sssMeanFreePath_,
                                         float3 pixelNormal,
                                         float3 sssNormal,
                                         float3 sssDistance, float bssrdfPDF, float intersectionPDF )
    {
        BssrdfDiffuseReflection d;
        d.bssrdfPDF = bssrdfPDF;
        d.intersectionPDF = intersectionPDF;
        d.pixelNormal = pixelNormal,
        d.sssMeanFreePath = sssMeanFreePath_;
        d.albedo = albedo_;
        //d.scatterDistance = d.sssMeanFreePath / sss_diffusion_profile_scatterDistance(d.albedo);
        
        d.DiffuseMeanFreePath = GetDiffuseMeanFreePathFromMeanFreePath( albedo_, sssMeanFreePath_ );
        float WorldUnitScale = 0.3f;
        float3 SSSRadius = GetMFPFromDMFPApprox(albedo_, albedo_, WorldUnitScale * d.DiffuseMeanFreePath);
        //sssDiffusionProfile = sss_diffusion_profile_scatterDistance( bsdf.data.diffuse );
        d.scatterDistance = SSSRadius; //bsdf.data.sssMeanFreePath / sssDiffusionProfile;
        
        
        d.sssDistance = sssDistance;
        return d;
    }
    
    float3 eval(const float3 wi, const float3 wo)
    {
        // need to fix bssrdfPDF, should used both normal vector of x1, x2
        //float bssrdfPDF = sss_sampling_disk_pdf(sssDistance, frame, frame.n, scatterDistance);

        const float3 diffusionProfile = sss_diffusion_profile_evaluate(length(sssDistance), scatterDistance);

        float3 bssrdf = albedo * diffusionProfile; // * disney_bssrdf_fresnel_evaluate(normal, v);

        float cosAtSurface = wo.z; // wo.z is dot(N,L)
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0,0,0);

        //float bsdf = disney_bssrdf_fresnel_evaluate(normalSample, l);

    /*
        return M_1_PI * bssrdf * cosAtSurface / ( bssrdfPDF * intersectionPDF );
    /*/
        return M_1_PI * bssrdf * cosAtSurface / ( bssrdfPDF );
    //*/
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        wo = sample_cosine_hemisphere_concentric(preGeneratedSample.xy, pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = float3(0,0,0);
            lobeP = 0.0;
            return false;
        }

        weight = albedo;
        lobeP = 1.0;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        // TODO:
        
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }
};

/** Disney's diffuse reflection.
    Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/
struct DiffuseReflectionDisney // : IBxDF
{
    float3 albedo;          ///< Diffuse albedo.
    float roughness;        ///< Roughness before remapping.

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0,0,0);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        wo = sample_cosine_hemisphere_concentric(preGeneratedSample.xy, pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = float3(0,0,0);
            lobeP = 0.0;
            return false;
        }

        weight = evalWeight(wi, wo);
        lobeP = 1.0;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float woDotH = dot(wo, h);
        float fd90 = 0.5f + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * wiScatter * woScatter;
    }
};

/** Frostbites's diffuse reflection.
    This is Disney's diffuse BRDF with an ad-hoc normalization factor to ensure energy conservation.
    Based on https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
*/
struct DiffuseReflectionFrostbite // : IBxDF
{
    float3 albedo;          ///< Diffuse albedo.
    float roughness;        ///< Roughness before remapping.

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0,0,0);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        wo = sample_cosine_hemisphere_concentric(preGeneratedSample.xy, pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = float3(0,0,0);
            lobeP = 0.0;
            return false;
        }

        weight = evalWeight(wi, wo);
        lobeP = 1.0;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float woDotH = dot(wo, h);
        float energyBias = lerp(0.f, 0.5f, roughness);
        float energyFactor = lerp(1.f, 1.f / 1.51f, roughness);
        float fd90 = energyBias + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * wiScatter * woScatter * energyFactor;
    }
};

/** Lambertian diffuse transmission.
*/
struct DiffuseTransmissionLambert // : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta) return float3(0,0,0);

        return M_1_PI * albedo * -wo.z;
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        wo = sample_cosine_hemisphere_concentric(preGeneratedSample.xy, pdf);
        wo.z = -wo.z;
        lobe = (uint)LobeType::DiffuseTransmission;

        if (min(wi.z, -wo.z) < kMinCosTheta)
        {
            weight = float3(0,0,0);
            lobeP = 0.0;
            return false;
        }

        weight = albedo;
        lobeP = 1.0;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * -wo.z;
    }
};

/** Specular reflection using microfacets.
*/
struct SpecularReflectionMicrofacet // : IBxDF
{
    float3 albedo;      ///< Specular albedo.
    float alpha;        ///< GGX width parameter.
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType.hlsli.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0,0,0);

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return float3(0,0,0);
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return float3(0,0,0);

        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, wo.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
#endif
        float3 F = evalFresnelSchlick(albedo, 1.f, wiDotH);
        return F * D * G * 0.25f / wi.z;
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        // Default initialization to avoid divergence at returns.
        wo = float3(0,0,0);
        weight = float3(0,0,0);
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;
        lobeP = 1.0;

        if (wi.z < kMinCosTheta) return false;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
        {
            if (!hasLobe(LobeType::DeltaReflection)) return false;

            wo = float3(-wi.x, -wi.y, wi.z);
            pdf = 0.f;
            weight = evalFresnelSchlick(albedo, 1.f, wi.z);
            lobe = (uint)LobeType::DeltaReflection;
            return true;
        }
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return false;

        // Sample the GGX distribution to find a microfacet normal (half vector).
#if GGXSampling == GGXSamplingVNDF
        float3 h = sampleGGX_VNDF(alpha, wi, preGeneratedSample.xy);    // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#elif GGXSampling == GGXSamplingBVNDF
        float3 h = sampleGGX_BVNDF(alpha, wi, preGeneratedSample.xy);
#elif GGXSampling == GGXSamplingNDF
        float3 h = sampleGGX_NDF(alpha, preGeneratedSample.xy);         // pdf = D(h) * h.z
#else
        #error unknown sampling type
#endif

        // Reflect the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z < kMinCosTheta) return false;

        pdf = evalPdf(wi, wo); // We used to have pdf returned as part of the sampleGGX_XXX functions but this made it easier to add bugs when changing due to code duplication in refraction cases
        weight = eval(wi, wo) / pdf;
        lobe = (uint)LobeType::SpecularReflection;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return 0.f;
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return 0.f;

        float3 h = normalize(wi + wo);

#if GGXSampling == GGXSamplingVNDF
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
#elif GGXSampling == GGXSamplingBVNDF
        float pdf = evalPdfGGX_BVNDF(alpha, wi, h);
#elif GGXSampling == GGXSamplingNDF
        float pdf = evalPdfGGX_NDF(alpha, wi, h);
#else
        #error unknown sampling type
#endif
        return pdf;
    }
};

/** Specular reflection and transmission using microfacets.
*/
struct SpecularReflectionTransmissionMicrofacet// : IBxDF
{
    float3 transmissionAlbedo;  ///< Transmission albedo.
    float alpha;                ///< GGX width parameter.
    float eta;                  ///< Relative index of refraction (etaI / etaT).
    uint activeLobes;           ///< BSDF lobes to include for sampling and evaluation. See LobeType.hlsli.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval(const float3 wi, const float3 wo)
    {
        if (min(wi.z, abs(wo.z)) < kMinCosTheta) return float3(0,0,0);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return float3(0,0,0);
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        const bool isReflection = wo.z > 0.f;
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return float3(0,0,0);

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wo + wi * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, abs(wo.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(wo.z));
#endif
        float F = evalFresnelDielectric(eta, wiDotH);

        if (isReflection)
        {
            return F * D * G * 0.25f / wi.z;
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            float t = eta * eta * wiDotH * woDotH / (wi.z * sqrtDenom * sqrtDenom);
            return transmissionAlbedo * (1.f - F) * D * G * abs(t);
        }
    }

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, float3 preGeneratedSample)
    {
        // Default initialization to avoid divergence at returns.
        wo = float3(0,0,0);
        weight = float3(0,0,0);
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;
        lobeP = 1;

        if (wi.z < kMinCosTheta) return false;

        // Get a random number to decide what lobe to sample.
        float lobeSample = preGeneratedSample.z;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f)
        {
            const bool hasReflection = hasLobe(LobeType::DeltaReflection);
            const bool hasTransmission = hasLobe(LobeType::DeltaTransmission);
            if (!(hasReflection || hasTransmission)) return false;

            float cosThetaT;
            float F = evalFresnelDielectric(eta, wi.z, cosThetaT);

            bool isReflection = hasReflection;
            if (hasReflection && hasTransmission)
            {
                isReflection = lobeSample < F;
                lobeP = (isReflection)?(F):(1-F);
            }
            else if (hasTransmission && F == 1.f)
            {
                return false;
            }

            pdf = 0.f;
            weight = isReflection ? float3(1,1,1) : transmissionAlbedo;
            if (!(hasReflection && hasTransmission)) weight *= float3( (isReflection ? F : 1.f - F).xxx );
            wo = isReflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cosThetaT);
            lobe = isReflection ? (uint)LobeType::DeltaReflection : (uint)LobeType::DeltaTransmission;

            if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection)) return false;

            return true;
        }
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if (!(hasReflection || hasTransmission)) return false;

        // Sample the GGX distribution of (visible) normals. This is our half vector.
#if GGXSampling == GGXSamplingVNDF
        float3 h = sampleGGX_VNDF(alpha, wi, preGeneratedSample.xy);    // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#elif GGXSampling == GGXSamplingBVNDF
        float3 h = sampleGGX_BVNDF(alpha, wi, preGeneratedSample.xy);
#elif GGXSampling == GGXSamplingNDF
        float3 h = sampleGGX_NDF(alpha, preGeneratedSample.xy);         // pdf = D(h) * h.z
#else
        #error unknown sampling type
#endif

        // Reflect/refract the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);

        float cosThetaT;
        float F = evalFresnelDielectric(eta, wiDotH, cosThetaT);

        bool isReflection = hasReflection;
        if (hasReflection && hasTransmission)
        {
            isReflection = lobeSample < F;
        }
        else if (hasTransmission && F == 1.f)
        {
            return false;
        }

        wo = isReflection ?
            (2.f * wiDotH * h - wi) :
            ((eta * wiDotH - cosThetaT) * h - eta * wi);

        if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection)) return false;

        float woDotH = dot(wo, h);

        lobe = isReflection ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

        pdf = evalPdf(wi, wo);  // <- this will have the correct Jacobian applied (for correct refraction pdf); We used to have pdf returned as part of the sampleGGX_XXX functions but this made it easier to add bugs when changing due to code duplication in refraction cases
        weight = pdf > 0.f ? eval(wi, wo) / pdf : float3(0, 0, 0);
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, abs(wo.z)) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return 0.f;
#endif

        bool isReflection = wo.z > 0.f;
        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return 0.f;

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wo + wi * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        float F = evalFresnelDielectric(eta, wiDotH);

#if GGXSampling == GGXSamplingVNDF
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
#elif GGXSampling == GGXSamplingBVNDF
        float pdf = evalPdfGGX_BVNDF(alpha, wi, h);
#elif GGXSampling == GGXSamplingNDF
        float pdf = evalPdfGGX_NDF(alpha, wi, h);
#else
        #error unknown sampling type
#endif
        if (isReflection)
        {   // Jacobian of the reflection operator.
            if (woDotH <= 0.f) return 0.f;
            pdf *= wiDotH / woDotH; 
        }
        else
        {   // Jacobian of the refraction operator.
            if (woDotH > 0.f) return 0.f;
            pdf *= wiDotH * 4.0f;
            float sqrtDenom = woDotH + eta * wiDotH;
            float denom = sqrtDenom * sqrtDenom;
            pdf *= abs(woDotH) / denom;
        }

        if (hasReflection && hasTransmission)
        {
            pdf *= isReflection ? F : 1.f - F;
        }

        return clamp( pdf, 0, FLT_MAX );
    }
};

// TODO: Reduce to 52B
/** BSDF parameters for the standard BSDF.
    These are needed for initializing a `FalcorBSDF` instance.
*/
struct StandardBSDFData
{
    float3 diffuse;                 ///< Diffuse albedo.
    float3 specular;                ///< Specular albedo.
    float roughness;                ///< This is the original roughness, before remapping.
    float metallic;                 ///< Metallic parameter, blends between dielectric and conducting BSDFs.
    float3 sssMeanFreePath;                  ///< Subsurface scattering mean free path.
    float eta;                      ///< Relative index of refraction (incident IoR / transmissive IoR).
    float3 transmission;            ///< Transmission color.
    float diffuseTransmission;      ///< Diffuse transmission, blends between diffuse reflection and transmission lobes.
    float specularTransmission;     ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.

    float3 sssPosition; ///< nearby position within SSS radius
    float3 position;
    float3 pixelNormal;
    float bssrdfPDF;
    float intersectionPDF;
    //float sssDistance;  ///< distance(position, sssPosition)

    static StandardBSDFData make() 
    { 
        StandardBSDFData d;
        d.diffuse = 0;
        d.specular = 0;
        d.roughness = 0;
        d.metallic = 0;
        d.eta = 0;
        d.sssMeanFreePath = 0;
        d.transmission = 0;
        d.diffuseTransmission = 0;
        d.specularTransmission = 0;
        d.sssPosition = 0;
        d.position = 0;
        d.pixelNormal = 0;
        d.bssrdfPDF = 1;
        d.intersectionPDF = 1;
        //d.sssDistance = 0;
        return d;
    }

    static StandardBSDFData make(
        float3 diffuse,
        float3 specular,
        float roughness,
        float metallic,
        float eta,
        float3 transmission,
        float diffuseTransmission,
        float specularTransmission,
        float3 pixelNormal,
        float3 position,
        float3 sssPosition,
        float bssrdf,
        float intersectionPDF
    )
    {
        StandardBSDFData d;
        d.diffuse = diffuse;
        d.specular = specular;
        d.roughness = roughness;
        d.metallic = metallic;
        d.eta = eta;
        d.transmission = transmission;
        d.diffuseTransmission = diffuseTransmission;
        d.specularTransmission = specularTransmission;
        return d;
    }
};

/** Mixed BSDF used for the standard material in Falcor.

    This consists of a diffuse and specular BRDF.
    A specular BSDF is mixed in using the specularTransmission parameter.
*/
struct FalcorBSDF // : IBxDF
{
#if DiffuseBrdf == DiffuseBrdfLambert
    DiffuseReflectionLambert diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfDisney
    DiffuseReflectionDisney diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfFrostbite
    DiffuseReflectionFrostbite diffuseReflection;
#endif
    DiffuseTransmissionLambert diffuseTransmission;

    SpecularReflectionMicrofacet specularReflection;
    SpecularReflectionTransmissionMicrofacet specularReflectionTransmission;

    float diffTrans;                        ///< Mix between diffuse BRDF and diffuse BTDF.
    float specTrans;                        ///< Mix between dielectric BRDF and specular BSDF.

    float pDiffuseReflection;               ///< Probability for sampling the diffuse BRDF.
    float pDiffuseTransmission;             ///< Probability for sampling the diffuse BTDF.
    float pSpecularReflection;              ///< Probability for sampling the specular BRDF.
    float pSpecularReflectionTransmission;  ///< Probability for sampling the specular BSDF.

    bool psdExclude; // disable PSD

    float3 _N;
    //float3 _T;
    //float3 _B;
    float3 sssNormal; // sss sample point's normal vector
    float3 sssMeanFreePath;
    float3 sssDistance; // sssPosition - position
    float bssrdfPDF;
    float intersectionPDF; // ~= 1.f/numInersections
    bool _isSss;
    
    bool isSss()
    {
        return _isSss > 0;
    }
    
    /** Initialize a new instance.
        \param[in] sd Shading data.
        \param[in] data BSDF parameters.
    */
    void __init(
        const MaterialHeader mtl,
        float3 N,
        float3 sssSampleNormal,
        //float3 T,
        //float3 B,
        float3 V,
        const StandardBSDFData data)
    {
        _N = N;
        //_T = T;
        //_B = B;
        sssNormal = sssSampleNormal;
        bssrdfPDF = data.bssrdfPDF;
        sssMeanFreePath = data.sssMeanFreePath;
        sssDistance = data.sssPosition - data.position;
        intersectionPDF = data.intersectionPDF;
        _isSss = any(sssMeanFreePath > 0.f);

        // TODO: Currently specular reflection and transmission lobes are not properly separated.
        // This leads to incorrect behaviour if only the specular reflection or transmission lobe is selected.
        // Things work fine as long as both or none are selected.

        // Use square root if we can assume the shaded object is intersected twice.
        float3 transmissionAlbedo = mtl.isThinSurface() ? data.transmission : sqrt(data.transmission);

        // Setup lobes.
        diffuseReflection.albedo = data.diffuse;
#if DiffuseBrdf != DiffuseBrdfLambert
        diffuseReflection.roughness = data.roughness;
#endif
        diffuseTransmission.albedo = transmissionAlbedo;

        // Compute GGX alpha.
        float alpha = data.roughness * data.roughness;
#if EnableDeltaBSDF
        // Alpha below min alpha value means using delta reflection/transmission.
        if (alpha < kMinGGXAlpha) alpha = 0.f;
#else
        alpha = max(alpha, kMinGGXAlpha);
#endif
        const uint activeLobes = mtl.getActiveLobes();

        psdExclude = mtl.isPSDExclude();

        specularReflection.albedo = data.specular;
        specularReflection.alpha = alpha;
        specularReflection.activeLobes = activeLobes;

        specularReflectionTransmission.transmissionAlbedo = transmissionAlbedo;
        // Transmission through rough interface with same IoR on both sides is not well defined, switch to delta lobe instead.
        specularReflectionTransmission.alpha = data.eta == 1.f ? 0.f : alpha;
        specularReflectionTransmission.eta = data.eta;
        specularReflectionTransmission.activeLobes = activeLobes;

        diffTrans = data.diffuseTransmission;
        specTrans = data.specularTransmission;

        // Compute sampling weights.
        float metallicBRDF = data.metallic * (1.f - specTrans);
        float dielectricBSDF = (1.f - data.metallic) * (1.f - specTrans);
        float specularBSDF = specTrans;

        float diffuseWeight = luminance(data.diffuse);
        float specularWeight = luminance(evalFresnelSchlick(data.specular, 1.f, dot(V, N)));

        pDiffuseReflection = (activeLobes & (uint)LobeType::DiffuseReflection) ? diffuseWeight * dielectricBSDF * (1.f - diffTrans) : 0.f;
        pDiffuseTransmission = (activeLobes & (uint)LobeType::DiffuseTransmission) ? diffuseWeight * dielectricBSDF * diffTrans : 0.f;
        pSpecularReflection = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection)) ? specularWeight * (metallicBRDF + dielectricBSDF) : 0.f;
        pSpecularReflectionTransmission = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection | (uint)LobeType::SpecularTransmission | (uint)LobeType::DeltaTransmission)) ? specularBSDF : 0.f;

        float normFactor = pDiffuseReflection + pDiffuseTransmission + pSpecularReflection + pSpecularReflectionTransmission;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pDiffuseTransmission *= normFactor;
            pSpecularReflection *= normFactor;
            pSpecularReflectionTransmission *= normFactor;
        }
    }
    
    /** Initialize a new instance.
    \param[in] sd Shading data.
    \param[in] data BSDF parameters.
*/
    void __init(const ShadingData shadingData, const StandardBSDFData data)
    {
        __init(shadingData.mtl, data.pixelNormal, shadingData.N, shadingData.V, data);
    }

    static FalcorBSDF make( const ShadingData shadingData, const StandardBSDFData data )     { FalcorBSDF ret; ret.__init(shadingData, data); return ret; }

    static FalcorBSDF make(
        const MaterialHeader mtl,
        float3 N,
        float3 V, 
        const StandardBSDFData data) 
    { 
        FalcorBSDF ret;
        ret.__init(mtl, data.pixelNormal, N, V, data); 
        return ret;
    }

    /** Returns the set of BSDF lobes.
        \param[in] data BSDF parameters.
        \return Returns a set of lobes (see LobeType.hlsli).
    */
    static uint getLobes(const StandardBSDFData data)
    {
#if EnableDeltaBSDF
        float alpha = data.roughness * data.roughness;
        bool isDelta = alpha < kMinGGXAlpha;
#else
        bool isDelta = false;
#endif
        float diffTrans = data.diffuseTransmission;
        float specTrans = data.specularTransmission;

        uint lobes = isDelta ? (uint)LobeType::DeltaReflection : (uint)LobeType::SpecularReflection;
        if (any(data.diffuse > 0.f) && specTrans < 1.f)
        {
            if (diffTrans < 1.f) lobes |= (uint)LobeType::DiffuseReflection;
            if (diffTrans > 0.f) lobes |= (uint)LobeType::DiffuseTransmission;
        }
        if (specTrans > 0.f) lobes |= (isDelta ? (uint)LobeType::DeltaTransmission : (uint)LobeType::SpecularTransmission);

        return lobes;
    }

#if RTXPT_DIFFUSE_SPECULAR_SPLIT
    void eval(const float3 wi, const float3 wo, out float3 diffuse, out float3 specular)
    {
        diffuse = 0.f; specular = 0.f;
        if (pDiffuseReflection > 0.f)
        {
            float3 diffuseReflectionEval = 0;
            if ( isSss() )
            {
                BssrdfDiffuseReflection bssrdfDiffuseReflection 
                    = BssrdfDiffuseReflection::make( diffuseReflection.albedo,
                                                     sssMeanFreePath,
                                                     _N,
                                                     sssNormal,
                                                     //_T,
                                                     //_B,
                                                     sssDistance, bssrdfPDF, intersectionPDF );
                diffuseReflectionEval = bssrdfDiffuseReflection.eval(wi, wo);
            }
            else
            {
                diffuseReflectionEval = diffuseReflection.eval(wi, wo);
            }
            diffuse += (1.f - specTrans) * (1.f - diffTrans) * diffuseReflectionEval;
            //if ( _isSss )
            //{
            //    diffuse = sssMeanFreePath;
            //}

        }
        if (pDiffuseTransmission > 0.f) diffuse += (1.f - specTrans) * diffTrans * diffuseTransmission.eval(wi, wo);
        if (pSpecularReflection > 0.f) specular += (1.f - specTrans) * specularReflection.eval(wi, wo);
        if (pSpecularReflectionTransmission > 0.f) specular += specTrans * (specularReflectionTransmission.eval(wi, wo));
    }
#else
    float3 eval(const float3 wi, const float3 wo)
    {
        float3 result = 0.f;
        if (pDiffuseReflection > 0.f) {
            float3 diffuseReflectionEval = 0;
            if ( isSss() )
            {
                BssrdfDiffuseReflection bssrdfDiffuseReflection
                    = BssrdfDiffuseReflection::make( diffuseReflection.albedo,
                                                     sssMeanFreePath,
                                                     _N,
                                                     sssNormal,
                                                     //_T,
                                                     //_B,
                                                     sssDistance );
                diffuseReflectionEval = bssrdfDiffuseReflection.eval( wi, wo );
            }
            else
            {
                diffuseReflectionEval = diffuseReflection.eval( wi, wo );
            }
            result += ( 1.f - specTrans ) * ( 1.f - diffTrans ) * diffuseReflectionEval;
        }
        if (pDiffuseTransmission > 0.f) result += (1.f - specTrans) * diffTrans * diffuseTransmission.eval(wi, wo);
        if (pSpecularReflection > 0.f) result += (1.f - specTrans) * specularReflection.eval(wi, wo);
        if (pSpecularReflectionTransmission > 0.f) result += specTrans * (specularReflectionTransmission.eval(wi, wo));
        return result;
    }
#endif

    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, out float lobeP, 
#if !RecycleSelectSamples
    float4 preGeneratedSample
#else
    float3 preGeneratedSample
#endif
    )
    {
        // Default initialization to avoid divergence at returns.
        wo = float3(0,0,0);
        weight = float3(0,0,0);
        pdf = 0.f;
        lobe = (uint)LobeType::DiffuseReflection;
        lobeP = 0.0;

        bool valid = false;
        float uSelect = preGeneratedSample.z;
#if !RecycleSelectSamples
        preGeneratedSample.z = preGeneratedSample.w;    // we've used .z for uSelect, shift left, .w is now unusable
#endif

        // Note: The commented-out pdf contributions below are always zero, so no need to compute them.

        if (uSelect < pDiffuseReflection)
        {
#if RecycleSelectSamples
            preGeneratedSample.z = clamp(uSelect / pDiffuseReflection, 0, OneMinusEpsilon); // note, this gets compiled out because bsdf below does not need .z, however it has been tested and can be used in case of a new bsdf that might require it
#endif
            
            valid = diffuseReflection.sample(wi, wo, pdf, weight, lobe, lobeP, preGeneratedSample.xyz);
            weight /= pDiffuseReflection;
            weight *= (1.f - specTrans) * (1.f - diffTrans);
            pdf *= pDiffuseReflection;
            lobeP *= pDiffuseReflection;
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission)
        {
            valid = diffuseTransmission.sample(wi, wo, pdf, weight, lobe, lobeP, preGeneratedSample.xyz);
            weight /= pDiffuseTransmission;
            weight *= (1.f - specTrans) * diffTrans;
            pdf *= pDiffuseTransmission;
            lobeP *= pDiffuseTransmission;
            // if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission + pSpecularReflection)
        {
#if RecycleSelectSamples
            preGeneratedSample.z = clamp((uSelect - (pDiffuseReflection + pDiffuseTransmission))/pSpecularReflection, 0, OneMinusEpsilon); // note, this gets compiled out because bsdf below does not need .z, however it has been tested and can be used in case of a new bsdf that might require it
#endif

            valid = specularReflection.sample(wi, wo, pdf, weight, lobe, lobeP, preGeneratedSample.xyz);
            weight /= pSpecularReflection;
            weight *= (1.f - specTrans);
            pdf *= pSpecularReflection;
            lobeP *= pSpecularReflection;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (pSpecularReflectionTransmission > 0.f)
        {
#if RecycleSelectSamples
            preGeneratedSample.z = clamp((uSelect - (pDiffuseReflection + pDiffuseTransmission + pSpecularReflection))/pSpecularReflectionTransmission, 0, OneMinusEpsilon);
#endif

            valid = specularReflectionTransmission.sample(wi, wo, pdf, weight, lobe, lobeP, preGeneratedSample.xyz);
            weight /= pSpecularReflectionTransmission;
            weight *= specTrans;
            pdf *= pSpecularReflectionTransmission;
            lobeP *= pSpecularReflectionTransmission;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        }

        if( !valid || (lobe & (uint)LobeType::Delta) != 0 )
            pdf = 0.0;

        return valid;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        float pdf = 0.f;
        if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
        if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
        if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        return pdf;
    }

    void evalDeltaLobes(const float3 wi, inout DeltaLobe deltaLobes[cMaxDeltaLobes], inout int deltaLobeCount, inout float nonDeltaPart)  // wi is in local space
    {
        deltaLobeCount = 2;             // currently - will be 1 more if we add clear coat :)
        for (int i = 0; i < deltaLobeCount; i++)
            deltaLobes[i] = DeltaLobe::make(); // init to zero
#if EnableDeltaBSDF == 0
#error not sure what to do here in this case
        return info;
#endif

        nonDeltaPart = pDiffuseReflection+pDiffuseTransmission;
        if ( specularReflection.alpha > 0 ) // if roughness > 0, lobe is not delta
            nonDeltaPart += pSpecularReflection;
        if ( specularReflectionTransmission.alpha > 0 ) // if roughness > 0, lobe is not delta
            nonDeltaPart += pSpecularReflectionTransmission;

        // no spec reflection or transmission? delta lobes are zero (we can just return, already initialized to 0)!
        if ( (pSpecularReflection+pSpecularReflectionTransmission) == 0 || psdExclude )    
            return;

        // note, deltaReflection here represents both this.specularReflection and this.specularReflectionTransmission's
        DeltaLobe deltaReflection, deltaTransmission;
        deltaReflection = deltaTransmission = DeltaLobe::make(); // init to zero
        deltaReflection.transmission    = false;
        deltaTransmission.transmission  = true;

        deltaReflection.dir  = float3(-wi.x, -wi.y, wi.z);

        if (specularReflection.alpha == 0 && specularReflection.hasLobe(LobeType::DeltaReflection))
        {
            deltaReflection.probability = pSpecularReflection;

            // re-compute correct thp for all channels (using float3 version of evalFresnelSchlick!) but then take out the portion that is handled by specularReflectionTransmission below!
            deltaReflection.thp = (1-pSpecularReflectionTransmission)*evalFresnelSchlick(specularReflection.albedo, 1.f, wi.z);
        }

        // Handle delta reflection/transmission.
        if (specularReflectionTransmission.alpha == 0.f)
        {
            const bool hasReflection = specularReflectionTransmission.hasLobe(LobeType::DeltaReflection);
            const bool hasTransmission = specularReflectionTransmission.hasLobe(LobeType::DeltaTransmission);
            if (hasReflection || hasTransmission)
            {
                float cosThetaT;
                float F = evalFresnelDielectric(specularReflectionTransmission.eta, wi.z, cosThetaT);

                if (hasReflection)
                {
                    float localProbability = pSpecularReflectionTransmission * F;
                    float3 weight = float3(1,1,1) * localProbability;
                    deltaReflection.thp += weight;
                    deltaReflection.probability += localProbability;
                }

                if (hasTransmission)
                {
                    float localProbability = pSpecularReflectionTransmission * (1.0-F);
                    float3 weight = specularReflectionTransmission.transmissionAlbedo * localProbability;
                    deltaTransmission.dir  = float3(-wi.x * specularReflectionTransmission.eta, -wi.y * specularReflectionTransmission.eta, -cosThetaT);
                    deltaTransmission.thp = weight;
                    deltaTransmission.probability = localProbability;
                }

                // 
                // if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection)) return false;
            }
        }

        // Lobes are by convention in this order, and the index must match BSDFSample::getDeltaLobeIndex() as well as the UI.
        // When we add clearcoat it goes after deltaReflection and so on.
        deltaLobes[0] = deltaTransmission;
        deltaLobes[1] = deltaReflection;
    }
};

#endif // __BxDF_HLSLI__