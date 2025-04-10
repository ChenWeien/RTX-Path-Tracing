/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_BRIDGE_DONUT_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_BRIDGE_DONUT_HLSLI__

// easier if we let Donut do this!
#define ENABLE_METAL_ROUGH_RECONSTRUCTION 1

#include "PathTracer/PathTracerBridge.hlsli"

#include "OpacityMicroMap/OmmDebug.hlsli"

// Donut-specific (native engine - we can include before PathTracer to avoid any collisions)
#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/vulkan.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/scene_material.hlsli>

#include "DonutBindings.hlsli"

enum DonutGeometryAttributes
{
    GeomAttr_Position       = 0x01,
    GeomAttr_TexCoord       = 0x02,
    GeomAttr_Normal         = 0x04,
    GeomAttr_Tangents       = 0x08,
    GeomAttr_PrevPosition   = 0x10,

    GeomAttr_All            = 0x1F
};

struct DonutGeometrySample
{
    InstanceData instance;
    GeometryData geometry;
    GeometryDebugData geometryDebug;
    MaterialConstants material;

    float3 vertexPositions[3];
    //float3 prevVertexPositions[3]; <- not needed for anything yet so we just use local variables and compute prevObjectSpacePosition
    float2 vertexTexcoords[3];

    float3 objectSpacePosition;
    float3 prevObjectSpacePosition;
    float2 texcoord;
    float3 flatNormal;
    float3 geometryNormal;
    float4 tangent;
    bool frontFacing;
};

float3 SafeNormalize(float3 input)
{
    float lenSq = dot(input,input);
    return input * rsqrt(max( 1.175494351e-38, lenSq));
}

DonutGeometrySample getGeometryFromHit(
    uint instanceIndex,
    uint geometryIndex,
    uint triangleIndex,
    float2 rayBarycentrics,
    DonutGeometryAttributes attributes,
    StructuredBuffer<InstanceData> instanceBuffer,
    StructuredBuffer<GeometryData> geometryBuffer,
    StructuredBuffer<GeometryDebugData> geometryDebugBuffer,
    StructuredBuffer<MaterialConstants> materialBuffer, 
    float3 rayDirection, 
    DebugContext debug)
{
    DonutGeometrySample gs = (DonutGeometrySample)0;

    gs.instance = instanceBuffer[instanceIndex];
    gs.geometry = geometryBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.geometryDebug = geometryDebugBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.material = materialBuffer[gs.geometry.materialIndex];
    
    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.vertexBufferIndex)];

    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);

    uint3 indices = indexBuffer.Load3(gs.geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);

    if (attributes & GeomAttr_Position)
    {
        gs.vertexPositions[0] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[0] * c_SizeOfPosition));
        gs.vertexPositions[1] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[1] * c_SizeOfPosition));
        gs.vertexPositions[2] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[2] * c_SizeOfPosition));
        gs.objectSpacePosition = interpolate(gs.vertexPositions, barycentrics);
    }

    if (attributes & GeomAttr_PrevPosition)
    {
        if( gs.geometry.prevPositionOffset != 0xFFFFFFFF )  // only present for skinned objects
        {
            float3 prevVertexPositions[3];
            /*gs.*/prevVertexPositions[0]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[0] * c_SizeOfPosition));
            /*gs.*/prevVertexPositions[1]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[1] * c_SizeOfPosition));
            /*gs.*/prevVertexPositions[2]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[2] * c_SizeOfPosition));
            gs.prevObjectSpacePosition  = interpolate(/*gs.*/prevVertexPositions, barycentrics);
        }
        else
            gs.prevObjectSpacePosition  = gs.objectSpacePosition;
    }

    if ((attributes & GeomAttr_TexCoord) && gs.geometry.texCoord1Offset != ~0u)
    {
        gs.vertexTexcoords[0] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        gs.vertexTexcoords[1] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        gs.vertexTexcoords[2] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
        gs.texcoord = interpolate(gs.vertexTexcoords, barycentrics);
    }

    if ((attributes & GeomAttr_Normal) && gs.geometry.normalOffset != ~0u)
    {
        float3 normals[3];

        normals[0] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + indices[0] * c_SizeOfNormal));
        normals[1] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + indices[1] * c_SizeOfNormal));
        normals[2] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + indices[2] * c_SizeOfNormal));
        gs.geometryNormal = interpolate(normals, barycentrics);
        gs.geometryNormal = mul(gs.instance.transform, float4(gs.geometryNormal, 0.0)).xyz;
        gs.geometryNormal = SafeNormalize(gs.geometryNormal);
    }

    if ((attributes & GeomAttr_Tangents) && gs.geometry.tangentOffset != ~0u)
    {
        float4 tangents[3];
        tangents[0] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + indices[0] * c_SizeOfTangent));
        tangents[1] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + indices[1] * c_SizeOfTangent));
        tangents[2] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + indices[2] * c_SizeOfTangent));

        gs.tangent.xyz = interpolate(tangents, barycentrics).xyz;
        gs.tangent.xyz = mul(gs.instance.transform, float4(gs.tangent.xyz, 0.0)).xyz;
        gs.tangent.xyz = SafeNormalize(gs.tangent.xyz);
        gs.tangent.w = tangents[0].w > 0 ? 1 : -1;
    }

    float3 objectSpaceFlatNormal = SafeNormalize(cross(
        gs.vertexPositions[1] - gs.vertexPositions[0],
        gs.vertexPositions[2] - gs.vertexPositions[0]));

    gs.flatNormal   = SafeNormalize(mul(gs.instance.transform, float4(objectSpaceFlatNormal, 0.0)).xyz);

    gs.frontFacing  = dot( -rayDirection, gs.flatNormal ) >= 0.0;

    return gs;
}

enum MaterialAttributes
{
    MatAttr_BaseColor    = 0x01,
    MatAttr_Emissive     = 0x02,
    MatAttr_Normal       = 0x04,
    MatAttr_MetalRough   = 0x08,
    MatAttr_Transmission = 0x10,
    MatAttr_Scatter      = 0x20,

    MatAttr_All          = 0x3F
};

float4 sampleTexture(uint textureIndexAndInfo, SamplerState samplerState, const ActiveTextureSampler textureSampler, float2 uv)
{
    uint textureIndex = textureIndexAndInfo & 0xFFFF;
    uint baseLOD = textureIndexAndInfo>>24;
    uint mipLevels = (textureIndexAndInfo>>16) & 0xFF;

    Texture2D tex2D = t_BindlessTextures[NonUniformResourceIndex(textureIndex)];

    return textureSampler.sampleTexture(tex2D, samplerState, uv, baseLOD, mipLevels);
}

// RLEye  RLchanged

static const float BlendMap2_Strength = 1; // 1 [ 0, 1 ]
static const float Shadow_Radius            = 0.242;//[ 0 1 ]
static const float Shadow_Hardness          = 0.715;//[ -1 1 ]
static const float Specular_Scale          = 0.8;// [ 0 10 ]
static const float Is_Left_Eye = 0; // 1 [ 0 1 ] checkbox
static const float3 Eye_Corner_Darkness_Color  = float3( 1, 0.7372, 0.70196 );
// group("Iris")
static const float Iris_Depth_Scale        = 1.3;  //[ 0  2.5 ]
static const float _Iris_Roughness          = 0; // 0  [ 0 1 ]
static const float Iris_Color_Brightness    = 0.598f;  //[ 0 5 ]
static const float Pupil_Scale              = 0.96;  //[ 0.91  1.1 ]
// subgroup("Iris_Advance")
static const float IoR                    = 1.4 ; // 1.4  [ 1 5 ]
static const float3 Iris_Cloudy_Color     = float3(0,0,0); // 0, 0, 0
static const float3 Iris_Inner_Color = float3(1,1,1); // 1 1 1
static const float Iris_Inner_Scale = 0;         // 0.000 [ 0 1 ]
static const float Iris_UV_Radius            = 0.145f; // [ 0.01   0.160001 ]
static const float3 Iris_Color = float3(1,1,1); // 1 1 1
// group("Limbus")
static const float Limbus_UV_Width_Color      = 0.037; // 0.037 [ 0  0.2 ]
static const float Limbus_Dark_Scale         = 6.6; // 6.600  [ 0  10 ]
// group("Sclera")
static const float ScleraBrightness        = 1; // 1 [ 0 5 ]
static const float Sclera_Roughness        = 0; // 0 [ 0  1 ]
// subgroup("Sclera_Advance")
static const float Sclera_Flatten_Normal   = 0.452; // 0.452 [ -5  5 ]
static const float Sclera_Normal_UV_Scale = 0.893; // 0.893 [ 0 5 ]
static const float Sclera_UV_Radius           = 0.930;   // 0.930 [ 0.01  1 ] Sclera_Scale

// group("Subsurface_Scatter")
// #SSS color1 ( 1, 1, 1 )
// #SSS slider0 5 [ 1, 5 ] spinMin 0.1 spinMax 100
//};

static const float Limbus_UV_Width_Shading = 0.2; // 0.3 [ 0  0.4 ]
static const float Iris_Concavity_Power = 0.5; // 0.5  [ 0.1  2 ]
static const float Iris_Concavity_Scale = 0.1116f; // 0.1116f  [ 0  4 ]
// float padding2
static const float Iris_Refraction_OnOff = 1; // 1.f  [ 0 1 ] checkbox
static const float Limbus_Pow = 5; // 5  [ 0  10 ]

float2 ScaleUVsByCenter( float2 uv, float ScaleReciprocal )
{
    float2 newUV = float2( 0.5, 0.5 ) + ( uv - float2( 0.5, 0.5 ) ) * ScaleReciprocal;
    return newUV;
}

// flatness = ( 1 - Weight )
float3 FlattenNormal( in float3 N, float flatness )
{
    return lerp( N, float3( 0, 0, 1 ), flatness );
}

float SphereMask( float2 uv, float2 Origin, float Radius, float Hardness )
{
    float2 uv2 = uv - Origin;
    float len = length( uv2 );
    len = ( len / Radius ) ; 
    len = ( 1 - len ) ; 
    float Softness = 1.f - Hardness;
    float mask = len / Softness; 
    mask = min ( max ( mask , 0 ) , 1 ) ; 
    return mask;
}

//CustomExpression0
float2 ML_EyeRefraction_IrisMask_Func ( float Iris_UV_Radius , float2 UV , float2 LimbusUVWidth ) 
{ 
    UV = UV - float2 ( 0.5f , 0.5f ) ; 
    float2 m , r ; 
    r = ( length ( UV ) - ( Iris_UV_Radius - LimbusUVWidth ) ) / LimbusUVWidth ; 
    m = saturate ( 1 - r ) ; 
    m = smoothstep ( 0 , 1 , m ) ; 
    return m ; 
}

float2 ML_EyeRefraction_IrisMask_Block(const DonutGeometrySample gs)
{
    float2 LimbusUVWidth = float2( Limbus_UV_Width_Color, Limbus_UV_Width_Shading );
    
    float2 uv = ScaleUVsByCenter( gs.texcoord, 1.f/Sclera_UV_Radius );

    return ML_EyeRefraction_IrisMask_Func( Iris_UV_Radius, uv, LimbusUVWidth );
}

// CustomExpression1
float3 RefractionDirection(float internalIoR , float3 normalW , float3 cameraW)
{
    float airIoR = 1.00029;
    
    float n = airIoR / internalIoR;
    
    float facing = dot(normalW, cameraW);
    
    float w = n * facing;
    
    float k = sqrt(1 + (w - n) * (w + n));
    
    float3 t;
    t = (w - k) * normalW - n * cameraW;
    t = normalize(t);
    return -t;
}

// A, B are unit vector
float3 GetPerpendicularUnitVector( float3 A, float3 B )
{
    float dotAB = dot( A, B );
    float3 projAonB = dotAB * B;
    float3 perpendicularA = A - projAonB;
    return normalize( perpendicularA );
}

float4 SampleTextureByIndex(const DonutGeometrySample gs, float2 UV, uint index, unsigned int Flag_UseTextureN, float4 defaultValue, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    if ((gs.material.flags & Flag_UseTextureN) != 0)
        return sampleTexture(index, materialSampler, textureSampler, UV);
    return defaultValue;
}

float3 SampleEyeScleraNormal(const DonutGeometrySample gs, float2 uv, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    // Sclera use normal texture slot
    float3 normalsTextureValue = SampleTextureByIndex(gs, uv, gs.material.normalTextureIndex, MaterialFlags_UseNormalTexture, float4(0,0,1,0), materialSampler, textureSampler).rgb;
    return UnpackNormalTexture(normalsTextureValue, 1);
}

float3 SampleEyeIrisNormal(const DonutGeometrySample gs, float2 uv, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    float3 normalsTextureValue = SampleTextureByIndex(gs, uv, gs.material.customTexture0Index, MaterialFlags_UseCustomTexture0, float4(0,0,1,0), materialSampler, textureSampler).rgb;
    return UnpackNormalTexture(normalsTextureValue, 1);
}

float3 SampleBaseColorTexture(float2 uv, const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    float3 TextureValue = SampleTextureByIndex(gs, uv, gs.material.baseOrDiffuseTextureIndex, MaterialFlags_UseBaseOrDiffuseTexture, float4(0,0,1,0), materialSampler, textureSampler).rgb;
    return TextureValue;
}

float AO_Block(const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    return 1;
}

float Roughness_Block(float2 ML_EyeRefraction_IrisMask, const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler )
{

    float3 OrmTextureValue = SampleTextureByIndex(gs, gs.texcoord, gs.material.metalRoughOrSpecularTextureIndex, MaterialFlags_UseMetalRoughOrSpecularTexture, float4(1,1,0,0), materialSampler, textureSampler).rgb;

    float roughness = gs.material.roughness * OrmTextureValue.g;
    return roughness + lerp( Sclera_Roughness, _Iris_Roughness, ML_EyeRefraction_IrisMask.r);
}

float Inner_Iris_Mask(float2 uv, const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    float TextureValue = SampleTextureByIndex(gs, uv, gs.material.customTexture1Index, MaterialFlags_UseCustomTexture1, float4(0,0,0,0), materialSampler, textureSampler).r;
    return TextureValue;
}

// Iris Albedo = albedo
// Iris Normal = gs.material.customTexture0Index
// Iris mask = gs.material.customTexture1
// Sclera Albedo = gs.material.customTexture2
// Sclera normal = gs.material.normalTextureIndex
//

float3 Sclera_Color_Block(const DonutGeometrySample gs, 
                          const SamplerState materialSampler, 
                          const ActiveTextureSampler textureSampler)
{
    float2 uv = gs.texcoord;
    float2 vScleraUv = ScaleUVsByCenter(uv, 1.f/Sclera_UV_Radius );
    if ( !(Is_Left_Eye > 0.f) )
    {
        vScleraUv = float2( 1, 1 ) - frac(vScleraUv);
    }

    float3 vScleraColor = SampleTextureByIndex(gs, uv, gs.material.customTexture2Index, MaterialFlags_UseCustomTexture2, float4(1,1,1,1), materialSampler, textureSampler).rgb;

    return ScleraBrightness * vScleraColor; // need sRGB to linear?
}

float3 Sclera_Normal_Block(const DonutGeometrySample gs, 
                           float2 ML_EyeRefraction_IrisMask, 
                           float3x3 TangentToWorld,
                           const SamplerState materialSampler,
                           const ActiveTextureSampler textureSampler )
{
    float2 vScleraNormalUv = ScaleUVsByCenter(gs.texcoord, 1.f/Sclera_Normal_UV_Scale ); 
    if ( ! ( Is_Left_Eye > 0.f ) )
    {
        vScleraNormalUv = float2( 1, 1 ) - frac( vScleraNormalUv );
    }
    float3 vTangentNormal = SampleEyeScleraNormal(gs, vScleraNormalUv, materialSampler, textureSampler );

    if ( ! ( Is_Left_Eye > 0.f ) )
    {
        vTangentNormal.y = -vTangentNormal.y;
        vTangentNormal.x = -vTangentNormal.x;
    }

    vTangentNormal = FlattenNormal( vTangentNormal, lerp ( Sclera_Flatten_Normal , 1 , ML_EyeRefraction_IrisMask . r ) );

    return TransformTangentVectorToWorld(TangentToWorld, vTangentNormal);
}

float3 Iris_Normal_Block(const DonutGeometrySample gs,
                         float3x3 TangentToWorld,
                         const SamplerState materialSampler,
                         const ActiveTextureSampler textureSampler )
{
    float3 vTangentNormal = SampleEyeIrisNormal(gs, gs.texcoord, materialSampler, textureSampler);
    return TransformTangentVectorToWorld(TangentToWorld, vTangentNormal);
}

#define DISPLACEMENT_MAP_SIZE 64
#define DISPLACEMENT_MAP_HALF_SIZE 32
#define DISPLACEMENT_MAX_IDX 22

float IrisDepthOffset(float UV)
{
    float Data[DISPLACEMENT_MAX_IDX+1] =
        {1.931, 1.922, 1.907, 1.887, 1.856, 1.815, 1.764, 1.705, 1.646, 1.591, 
         1.537, 1.486, 1.436, 1.379, 1.318, 1.253, 1.179, 1.089, 0.998, 0.886, 
         0.748, 0.573, 0.267};
    float UVcenteredX = UV;
    float Radius = abs(UVcenteredX);
    float ScaledRadius = Radius * (DISPLACEMENT_MAP_SIZE-1);

    if (ScaledRadius > DISPLACEMENT_MAX_IDX)
        return 0;

    int LowerIndex = int(floor(ScaledRadius));
    int UpperIndex = int(ceil(ScaledRadius));

    LowerIndex = clamp(LowerIndex, 0, DISPLACEMENT_MAX_IDX);
    UpperIndex = clamp(UpperIndex, 0, DISPLACEMENT_MAX_IDX);

    float LowerValue = Data[LowerIndex];
    float UpperValue = Data[UpperIndex];
    float Interpolation = ScaledRadius - float(LowerIndex);
    return lerp(LowerValue, UpperValue, Interpolation);
}

float IrisDisplacementMap( float2 UV )
{
    float Data[DISPLACEMENT_MAX_IDX+1] =
        {1.931, 1.922, 1.907, 1.887, 1.856, 1.815, 1.764, 1.705, 1.646, 1.591, 
         1.537, 1.486, 1.436, 1.379, 1.318, 1.253, 1.179, 1.089, 0.998, 0.886, 
         0.748, 0.573, 0.267};

    // [0,1] => [-0.5, 0.5] *2 => [-1, 1]
    float2 UVcentered = clamp( 2 * (frac(UV) - float2(0.5f, 0.5f)), -1, 1 );
    float Radius = sqrt(UVcentered.x * UVcentered.x + UVcentered.y * UVcentered.y);

    float ScaledRadius = Radius * (DISPLACEMENT_MAP_HALF_SIZE-1);

    if ( ScaledRadius > DISPLACEMENT_MAX_IDX ) return 0;

    // Calculate the two nearest indices
    int LowerIndex = int(floor(ScaledRadius));
    int UpperIndex = int(ceil(ScaledRadius));

    LowerIndex = clamp(LowerIndex, 0, DISPLACEMENT_MAX_IDX);
    UpperIndex = clamp(UpperIndex, 0, DISPLACEMENT_MAX_IDX);

    float LowerValue = Data[LowerIndex];
    float UpperValue = Data[UpperIndex];
    float Interpolation = ScaledRadius - float(LowerIndex);

    return lerp(LowerValue, UpperValue, Interpolation);
}

float Iris_Depth_Block()
{
    return IrisDepthOffset(Sclera_UV_Radius * Iris_UV_Radius);
}

//!! Parameters.WorldNormal must be set
float2 Derive_Tangents_Block(float3 EyeDirectionWorld, float3 WorldNormal, float3 CameraVector, float3x3 TangentToWorld, const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    float3 vRefractDir = RefractionDirection(IoR , WorldNormal , CameraVector);

    float Input_DepthPlaneOffSet = Iris_Depth_Block() ; 
    float MidPlaneDisplacement = IrisDisplacementMap(gs.texcoord);

    float3 IrisDepth = (float3)( Iris_Depth_Scale * max( MidPlaneDisplacement - Input_DepthPlaneOffSet, 0 ) ) ;

    float dotVE = dot(CameraVector , EyeDirectionWorld);
    float dotVEsq = (dotVE * dotVE);

    float3 Scale_Refrected_Offset_Dir = vRefractDir * IrisDepth / lerp ( 0.325f , 1 , dotVEsq ) ; 

    float3 TangentBasisUnitX = normalize(TangentToWorld[0]);

    float3 vPerp = GetPerpendicularUnitVector( TangentBasisUnitX, EyeDirectionWorld );

    float dotPR = dot ( vPerp , Scale_Refrected_Offset_Dir ) ; 
    float3 crossEP = cross ( EyeDirectionWorld, vPerp ) ; // UE is left-handed, swap order
    float dotCR = dot ( crossEP , Scale_Refrected_Offset_Dir ) ; 
    return float2 ( dotPR , dotCR );
}

float2  ScalePupils ( float2 UV , float PupilScale, float fDist ) 
{ 
    // Scale UVs from from unit circle in or out from center
    float2 UVcentered = UV - float2 ( 0.5f , 0.5f ) ; 
    float UVlength = length ( UVcentered ) ; 
    // UV on circle at distance 0.5 from the center, in direction of original UV
    float2 UVmax = normalize ( UVcentered ) * fDist ; 
    
    float2 UVscaled = lerp ( UVmax , float2 ( 0.f , 0.f ) , saturate ( ( 1.f - UVlength * 0.5f ) * PupilScale ) ) ; 
    return UVscaled + float2 ( 0.5f , 0.5f ) ; 
}

float2 Pupil_Size_Block( float2 UV, float2 ML_EyeRefraction_RefractedUV )
{
    float2 uv = ScaleUVsByCenter( UV, 1.f/Sclera_UV_Radius );
    //  UVs with Refraction or not
    uv = lerp ( uv , ML_EyeRefraction_RefractedUV , Iris_Refraction_OnOff ) ; 
    uv = ( uv - 0.5f ) ; 
    uv = ( uv * 1 / ( Iris_UV_Radius * 2.f )  ) ; 
    uv = ( uv + 0.5f ) ; 

    // Scale Pupils
    return ScalePupils ( uv , Pupil_Scale, 0.5f ) ; 
}

float3 Eye_Ball_Corner_Darkness_Block(float2 vScleraScaledUv)
{
    float fMask = SphereMask(vScleraScaledUv, float2(0.5, 0.5), Shadow_Radius, Shadow_Hardness);

    return lerp(Eye_Corner_Darkness_Color, float3(1, 1, 1), fMask);
}

float PositiveClampedPow( float X, float Y )
{
    return pow( max( X, 0.0f ), Y );
}

float Iris_Distance_Block( float2 ML_EyeRefraction_RefractedUV )
{
    float2 uv = ML_EyeRefraction_RefractedUV - (float2)0.5; 
    float len = length( uv );
    len = Iris_Concavity_Scale * ( len / Iris_UV_Radius ) ; 
    return PositiveClampedPow ( len , Iris_Concavity_Power ) ; 
}

float Limbus_Color_Block( float2 vScalePupils )
{
    float2 v = ( vScalePupils - float2(0.5f, 0.5f) ) * Limbus_Dark_Scale;

    return 1 - PositiveClampedPow ( length( v ) , Limbus_Pow ) ; 
}

float2 ML_EyeRefraction_RefractedUV_Func(
            float2 ML_EyeRefraction_IrisMask,
            float3 EyeDirectionWorld,
            float3 WorldNormal,
            float3 CameraVector,
            float3x3 TangentToWorld,
            const DonutGeometrySample gs, const SamplerState materialSampler, const ActiveTextureSampler textureSampler )
{
    float2 vScaledUv = ScaleUVsByCenter(gs.texcoord, 1.f/Sclera_UV_Radius ); 

    // Derive Tangent
    float2 vRefractedUVOffset = Derive_Tangents_Block(EyeDirectionWorld, WorldNormal, CameraVector, TangentToWorld, gs, materialSampler, textureSampler);

    // Scale offset to within Iris
    float2 vOffset = (float2(-Iris_UV_Radius, Iris_UV_Radius) * vRefractedUVOffset);

    float2 vRefractedUv = (vOffset + vScaledUv);
    // Use Refracted UV within Iris based on Iris Mask
    return lerp(vScaledUv, vRefractedUv, ML_EyeRefraction_IrisMask.r);
}



float3 Iris_Color_Block( float2 ML_EyeRefraction_IrisMask,
                         float2 ML_EyeRefraction_RefractedUV,
                         const DonutGeometrySample gs,
                         const SamplerState materialSampler, const ActiveTextureSampler textureSampler )
{
    float2 UV = gs.texcoord;
    float2 vScleraScaledUv = ScaleUVsByCenter(UV, 1.f/Sclera_UV_Radius );

    // Scale Pupils
    float2 vScalePupils = Pupil_Size_Block(UV, ML_EyeRefraction_RefractedUV ); 

    // Iris Color
    float2 vPupilUV = ScalePupils( vScalePupils , lerp( 0.92f, 0.952f, Iris_Inner_Scale ), 4 );
    
    //return float3(vPupilUV,0);
    
    float fLerpFactor = Inner_Iris_Mask( vPupilUV, gs, materialSampler, textureSampler );
    
    return (float3)fLerpFactor;
    
    float3 vIrisColorLerp = lerp( Iris_Color * Iris_Color_Brightness, Iris_Inner_Color, fLerpFactor );

    float3 vBaseColor = SampleBaseColorTexture( vScalePupils, gs, materialSampler, textureSampler );

    vBaseColor *= ML_EyeRefraction_IrisMask.r * Iris_Color_Brightness; // yes Iris_Color_Brightness is multiplied here to compatible with CP1

    vBaseColor = vBaseColor * vIrisColorLerp * Limbus_Color_Block( vScalePupils );

    float3 vScleraColor = Sclera_Color_Block(gs, materialSampler, textureSampler);

    vBaseColor = lerp ( vScleraColor , vBaseColor , float ( ML_EyeRefraction_IrisMask . r ) ) ; 

    float3 vIrisCloudyColor = Iris_Cloudy_Color * SphereMask( vScalePupils, float2( 0.5f, 0.5f ), 0.18f, 0.2f );

    vBaseColor += vIrisCloudyColor;

    float3 vCornerDarkness = Eye_Ball_Corner_Darkness_Block( vScleraScaledUv ) ; 

    return vBaseColor * vCornerDarkness;
}

MaterialSample sampleGeometryMaterialEye(float3 rayDir, float3x3 TBN, const DonutGeometrySample gs, const MaterialAttributes attributes, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    MaterialTextureSample textures = DefaultMaterialTextures();

    float3x3 TangentToWorld = TBN;

    float2 ML_EyeRefraction_IrisMask = ML_EyeRefraction_IrisMask_Block(gs);
    float3 WorldNormal = Sclera_Normal_Block(gs, ML_EyeRefraction_IrisMask, TangentToWorld, materialSampler, textureSampler);
    float3 vIrisWorldNormal = Iris_Normal_Block(gs, TangentToWorld, materialSampler, textureSampler);
    float3 EyeDirectionWorld = vIrisWorldNormal;

    float3 CameraVector = -rayDir; // float3 vView = normalize( g_vCameraPos - Input.vPosition.xyz );
    
    float2 ML_EyeRefraction_RefractedUV = ML_EyeRefraction_RefractedUV_Func(
             ML_EyeRefraction_IrisMask,
             EyeDirectionWorld,
             WorldNormal,
             CameraVector,
             TangentToWorld,
            gs, materialSampler, textureSampler );
    
    
    // Scale Pupils
    float2 vScalePupils = Pupil_Size_Block(gs.texcoord, ML_EyeRefraction_RefractedUV);
    
    float3 vBaseColor = Iris_Color_Block( ML_EyeRefraction_IrisMask, 
                                          ML_EyeRefraction_RefractedUV,
                                          gs, materialSampler, textureSampler);
    
    float IrisMask = saturate( ML_EyeRefraction_IrisMask.g );
    float IrisDistance = saturate( Iris_Distance_Block( ML_EyeRefraction_RefractedUV ) );
    
    MaterialSample result = (MaterialSample)0;
    result.shadingNormal = WorldNormal;
    result.geometryNormal = vIrisWorldNormal; //gs.geometryNormal;
    result.diffuseAlbedo = result.baseColor = vBaseColor;
    //float3(vScalePupils, 0 ); //float3(IrisDistance,IrisDistance,IrisDistance);//ML_EyeRefraction_IrisMask, 0); //ML_EyeRefraction_RefractedUV, 0 ); //vBaseColor;
    result.ior = gs.material.ior;

    result.irisMask = IrisMask;
    result.irisNormal = vIrisWorldNormal;
    const float3 CausticNormal = normalize(lerp(vIrisWorldNormal, -WorldNormal, IrisMask * IrisDistance));
    result.causticNormal = CausticNormal;

    result.shadowNoLFadeout = gs.material.shadowNoLFadeout;
    result.metalness = 0;
    result.opacity = 1;
    result.occlusion = 1;
    result.roughness = Roughness_Block(ML_EyeRefraction_IrisMask, gs, materialSampler, textureSampler );
    result.modelId = MODELID_EYE;
    return result;
}


MaterialSample sampleGeometryMaterial(float3 rayDir, float3x3 TBN, uniform PathTracer::OptimizationHints optimizationHints, const DonutGeometrySample gs, const MaterialAttributes attributes, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{

  #if (defined(RLSHADER) && RLSHADER==Eye)
    return sampleGeometryMaterialEye(rayDir, TBN, gs, attributes, materialSampler, textureSampler);
  #endif

    MaterialTextureSample textures = DefaultMaterialTextures();

    if( !optimizationHints.NoTextures )
    {
        if ((attributes & MatAttr_BaseColor) && (gs.material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
            textures.baseOrDiffuse = sampleTexture(gs.material.baseOrDiffuseTextureIndex, materialSampler, textureSampler, gs.texcoord);

        if ((attributes & MatAttr_Emissive) && (gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0)
            textures.emissive = sampleTexture(gs.material.emissiveTextureIndex, materialSampler, textureSampler, gs.texcoord);
    
        if ((attributes & MatAttr_Normal) && (gs.material.flags & MaterialFlags_UseNormalTexture) != 0)
            textures.normal = sampleTexture(gs.material.normalTextureIndex, materialSampler, textureSampler, gs.texcoord);

        if ((attributes & MatAttr_MetalRough) && (gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
            textures.metalRoughOrSpecular = sampleTexture(gs.material.metalRoughOrSpecularTextureIndex, materialSampler, textureSampler, gs.texcoord);

        if( !optimizationHints.NoTransmission )
        {
            if ((attributes & MatAttr_Transmission) && (gs.material.flags & MaterialFlags_UseTransmissionTexture) != 0)
                textures.transmission = sampleTexture(gs.material.transmissionTextureIndex, materialSampler, textureSampler, gs.texcoord);
        }

        if ( ( attributes & MatAttr_Scatter ) && ( gs.material.flags & MaterialFlags_UseScatterTexture ) != 0 )
            textures.scatter = sampleTexture( gs.material.scatterTextureIndex, materialSampler, textureSampler, gs.texcoord );

        if ((gs.material.flags & MaterialFlags_UseCustomTexture0) != 0)
            textures.custom0 = sampleTexture(gs.material.customTexture0Index, materialSampler, textureSampler, gs.texcoord);
        if ((gs.material.flags & MaterialFlags_UseCustomTexture1) != 0)
            textures.custom1 = sampleTexture(gs.material.customTexture1Index, materialSampler, textureSampler, gs.texcoord);
        if ((gs.material.flags & MaterialFlags_UseCustomTexture2) != 0)
            textures.custom2 = sampleTexture(gs.material.customTexture2Index, materialSampler, textureSampler, gs.texcoord);
        if ((gs.material.flags & MaterialFlags_UseCustomTexture3) != 0)
            textures.custom3 = sampleTexture(gs.material.customTexture3Index, materialSampler, textureSampler, gs.texcoord);
    }
    return EvaluateSceneMaterial(gs.geometryNormal, gs.tangent, gs.material, textures);
}

static OpacityMicroMapDebugInfo loadOmmDebugInfo(const DonutGeometrySample donutGS, const uint triangleIndex, const TriangleHit triangleHit)
{
    OpacityMicroMapDebugInfo ommDebug = OpacityMicroMapDebugInfo::initDefault();

#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS
    if (donutGS.geometryDebug.ommIndexBufferIndex != -1 &&
        donutGS.geometryDebug.ommIndexBufferOffset != 0xFFFFFFFF)
    {
        ByteAddressBuffer ommIndexBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommIndexBufferIndex)];
        ByteAddressBuffer ommDescArrayBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommDescArrayBufferIndex)];
        ByteAddressBuffer ommArrayDataBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommArrayDataBufferIndex)];

        OpacityMicroMapContext ommContext = OpacityMicroMapContext::make(
            ommIndexBuffer, donutGS.geometryDebug.ommIndexBufferOffset, donutGS.geometryDebug.ommIndexBuffer16Bit,
            ommDescArrayBuffer, donutGS.geometryDebug.ommDescArrayBufferOffset,
            ommArrayDataBuffer, donutGS.geometryDebug.ommArrayDataBufferOffset,
            triangleIndex,
            triangleHit.barycentrics.xy
        );

        ommDebug.hasOmmAttachment = true;
        ommDebug.opacityStateDebugColor = OpacityMicroMapDebugViz(ommContext);
    }
#endif

    return ommDebug;
}

static void surfaceDebugViz(const uniform PathTracer::OptimizationHints optimizationHints, const PathTracer::SurfaceData surfaceData, const TriangleHit triangleHit, const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const OpacityMicroMapDebugInfo ommDebug, DebugContext debug)
{
#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS
    if (g_Const.debug.debugViewType == (int)DebugViewType::Disabled || pathVertexIndex != 1)
        return;

    //const VertexData vd     = surfaceData.vd;
    const ShadingData shadingData = surfaceData.shadingData;
    const ActiveBSDF bsdf = surfaceData.bsdf;

    // these work only when ActiveBSDF is StandardBSDF - make an #ifdef if/when this becomes a problem
    StandardBSDFData bsdfData = bsdf.data;

    switch (g_Const.debug.debugViewType)
    {
    case ((int)DebugViewType::FirstHitBarycentrics):                debug.DrawDebugViz(float4(triangleHit.barycentrics, 0.0, 1.0)); break;
    case ((int)DebugViewType::FirstHitFaceNormal):                  debug.DrawDebugViz(float4(DbgShowNormalSRGB(shadingData.frontFacing ? shadingData.faceN : -shadingData.faceN), 1.0)); break;
    case ((int)DebugViewType::FirstHitShadingNormal):               debug.DrawDebugViz(float4(DbgShowNormalSRGB(shadingData.N), 1.0)); break;
    case ((int)DebugViewType::FirstHitShadingTangent):              debug.DrawDebugViz(float4(DbgShowNormalSRGB(shadingData.T), 1.0)); break;
    case ((int)DebugViewType::FirstHitShadingBitangent):            debug.DrawDebugViz(float4(DbgShowNormalSRGB(shadingData.B), 1.0)); break;
    case ((int)DebugViewType::FirstHitFrontFacing):                 debug.DrawDebugViz(float4(saturate(float3(0.15, 0.1 + shadingData.frontFacing, 0.15)), 1.0)); break;
    case ((int)DebugViewType::FirstHitThinSurface):                 debug.DrawDebugViz(float4(saturate(float3(0.15, 0.1 + shadingData.mtl.isThinSurface(), 0.15)), 1.0)); break;
    case ((int)DebugViewType::FirstHitShaderPermutation):           debug.DrawDebugViz(float4(optimizationHints.NoTextures, optimizationHints.NoTransmission, optimizationHints.OnlyDeltaLobes, 1.0)); break;
    case ((int)DebugViewType::FirstHitDiffuse):                     debug.DrawDebugViz(float4(bsdfData.diffuse.xyz, 1.0)); break;
    case ((int)DebugViewType::FirstHitSpecular):                    debug.DrawDebugViz(float4(bsdfData.specular.xyz, 1.0)); break;
    case ((int)DebugViewType::FirstHitRoughness):                   debug.DrawDebugViz(float4(bsdfData.roughness.xxx, 1.0)); break;
    case ((int)DebugViewType::FirstHitMetallic):                    debug.DrawDebugViz(float4(bsdfData.metallic.xxx, 1.0)); break;
    case ((int)DebugViewType::FirstHitOpacityMicroMapOverlay):      debug.DrawDebugViz(float4(ommDebug.opacityStateDebugColor, 1.0)); break;
    default: break;
    }
#endif
}

uint Bridge::getSampleBaseIndex()
{
    return g_Const.ptConsts.sampleBaseIndex;
}

uint Bridge::getSubSampleCount()
{
#if PATH_TRACER_MODE != PATH_TRACER_MODE_BUILD_STABLE_PLANES
    return g_Const.ptConsts.subSampleCount;
#else
    return 1.0;
#endif
}

float Bridge::getNoisyRadianceAttenuation()
{
    // When using multiple samples within pixel in realtime mode (which share identical camera ray), only noisy part of radiance (i.e. not direct sky) needs to be attenuated!
#if PATH_TRACER_MODE != PATH_TRACER_MODE_BUILD_STABLE_PLANES
    return g_Const.ptConsts.invSubSampleCount;
#else
    return 1.0;
#endif
}

uint Bridge::getMaxBounceLimit()
{
    return g_Const.ptConsts.bounceCount;
}

uint Bridge::getMaxDiffuseBounceLimit()
{
    return g_Const.ptConsts.diffuseBounceCount;
}

// note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
Ray Bridge::computeCameraRay(const uint2 pixelPos, const uint subSampleIndex)
{
    SampleGenerator sampleGenerator = SampleGenerator::make(pixelPos, 0, Bridge::getSampleBaseIndex() + subSampleIndex);

    // compute camera ray! would make sense to compile out if unused
    float2 subPixelOffset;
    if (g_Const.ptConsts.enablePerPixelJitterAA)
        subPixelOffset = sampleNext2D(sampleGenerator) - 0.5.xx;
    else
        subPixelOffset = g_Const.ptConsts.camera.jitter * float2(1, -1); // conversion so that ComputeRayThinlens matches Donut offset convention in View.cpp->UpdateCache()
    const float2 cameraDoFSample = sampleNext2D(sampleGenerator);
    //return ComputeRayPinhole( g_Const.ptConsts.camera, pixelPos, subPixelOffset );
    Ray ray = ComputeRayThinlens( g_Const.ptConsts.camera, pixelPos, subPixelOffset, cameraDoFSample ); 

#if 0  // fallback: use inverted matrix) useful for correctness validation; with DoF disabled (apertureRadius/focalDistance == near zero), should provide same rays as above code - otherwise something's really broken
    PlanarViewConstants view = g_Const.view;
    float2 uv = (float2(pixelPos) + 0.5) * view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1e-7, 1);
    float4 worldPos = mul(clipPos, view.matClipToWorld);
    worldPos.xyz /= worldPos.w;
        
    ray.origin  = view.cameraDirectionOrPosition.xyz;
    ray.dir     = normalize(worldPos.xyz - ray.origin);
#endif
    return ray;
}

/** Helper to create a texture sampler instance.
The method for computing texture level-of-detail depends on the configuration.
\param[in] path Path state.
\param[in] isPrimaryTriangleHit True if primary hit on a triangle.
\return Texture sampler instance.
*/
ActiveTextureSampler Bridge::createTextureSampler(const RayCone rayCone, const float3 rayDir, float coneTexLODValue, float3 normalW, bool isPrimaryHit, bool isTriangleHit, float texLODBias)
{
#if ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_EXPLICIT
    return ExplicitLodTextureSampler::make(texLODBias);
#elif ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_RAY_CONES
    float lambda = rayCone.computeLOD(coneTexLODValue, rayDir, normalW, true);
    lambda += texLODBias;
    return ExplicitRayConesLodTextureSampler::make(lambda);
#endif
}

void Bridge::loadSurfacePosNormOnly(out float3 posW, out float3 faceN, const TriangleHit triangleHit, DebugContext debug)
{
    const uint instanceIndex    = triangleHit.instanceID.getInstanceIndex();
    const uint geometryIndex    = triangleHit.instanceID.getGeometryIndex();
    const uint triangleIndex    = triangleHit.primitiveIndex;
    DonutGeometrySample donutGS = getGeometryFromHit(instanceIndex, geometryIndex, triangleIndex, triangleHit.barycentrics, GeomAttr_Position,
        t_InstanceData, t_GeometryData, t_GeometryDebugData, t_MaterialConstants, float3(0,0,0), debug);
    posW    = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;
    faceN   = donutGS.flatNormal;
}

PathTracer::SurfaceData Bridge::loadSurface(const uniform PathTracer::OptimizationHints optimizationHints, const TriangleHit triangleHit, const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, DebugContext debug)
{
    const bool isPrimaryHit     = pathVertexIndex == 1;
    const uint instanceIndex    = triangleHit.instanceID.getInstanceIndex();
    const uint geometryIndex    = triangleHit.instanceID.getGeometryIndex();
    const uint triangleIndex    = triangleHit.primitiveIndex;

    DonutGeometrySample donutGS = getGeometryFromHit(instanceIndex, geometryIndex, triangleIndex, triangleHit.barycentrics, GeomAttr_TexCoord | GeomAttr_Position | GeomAttr_Normal | GeomAttr_Tangents | GeomAttr_PrevPosition,
        t_InstanceData, t_GeometryData, t_GeometryDebugData, t_MaterialConstants, rayDir, debug);

    // Convert Donut to RTXPT vertex data
    VertexData ptVertex;
    ptVertex.posW           = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;
    float3 prevPosW             = mul(donutGS.instance.prevTransform, float4(donutGS.prevObjectSpacePosition, 1.0)).xyz;
    ptVertex.normalW        = donutGS.geometryNormal;     // this normal is not guaranteed to point towards the viewer (but shading normal will get corrected below)
    ptVertex.tangentW       = donutGS.tangent;            // .w holds the sign/direction for the bitangent
    ptVertex.texC           = donutGS.texcoord;
    ptVertex.faceNormalW    = donutGS.flatNormal;         // this normal is not guaranteed to point towards the viewer (but shading normal will get corrected below)
    ptVertex.curveRadius    = 1;                          // unused for triangle meshes
        
    // transpose is to go from Donut row_major to Falcor column_major; it is likely unnecessary here since both should work the same for this specific function, but leaving in for correctness
    ptVertex.coneTexLODValue= computeRayConeTriangleLODValue( donutGS.vertexPositions, donutGS.vertexTexcoords, transpose((float3x3)donutGS.instance.transform) );

    // using flat (triangle) normal makes more sense since actual triangle surface is where the textures are sampled on (plus geometry normals are borked in some datasets)
    ActiveTextureSampler textureSampler = createTextureSampler(rayCone, rayDir, ptVertex.coneTexLODValue, donutGS.flatNormal/*donutGS.geometryNormal*/, isPrimaryHit, true, g_Const.ptConsts.texLODBias);

    // See MaterialFactory.hlsli in Falcor
    ShadingData ptShadingData = ShadingData::make();

    ptShadingData.posW = ptVertex.posW;
    ptShadingData.uv   = ptVertex.texC;
    ptShadingData.V    = -rayDir;
    ptShadingData.N    = ptVertex.normalW;

    // after this point we have valid tangent space in ptShadingData.N/.T/.B using geometry (interpolated) normal, but without normalmap yet
    const bool validTangentSpace = computeTangentSpace(ptShadingData, ptVertex.tangentW);

    // Primitive data
    ptShadingData.faceN = ptVertex.faceNormalW;         // must happen before adjustShadingNormal!
    ptShadingData.vertexN = (donutGS.frontFacing)?(donutGS.geometryNormal):(-donutGS.geometryNormal);
    ptShadingData.frontFacing = donutGS.frontFacing;        // must happen before adjustShadingNormal!
    ptShadingData.curveRadius = ptVertex.curveRadius;

    float3 CameraVector = normalize(g_Const.view.cameraDirectionOrPosition.xyz - ptShadingData.posW);
    float3x3 TBN = { ptShadingData.T, ptShadingData.B, ptShadingData.N };
    
    // Get donut material (normal map is evaluated here)
    MaterialSample donutMaterial = sampleGeometryMaterial(-CameraVector, TBN, optimizationHints, donutGS, MatAttr_All, s_MaterialSampler, textureSampler);

    ptShadingData.N = donutMaterial.shadingNormal;

    // Donut -> Falcor
    //const bool donutMaterialDoubleSided = (donutGS.material.flags & MaterialFlags_DoubleSided) != 0;  // removed; all triangles are double sided to avoid breaking path tracing logic in various cases
    const bool donutMaterialThinSurface = (donutGS.material.flags & MaterialFlags_ThinSurface) != 0;
    //const bool alphaTested = (donutGS.material.domain == MaterialDomain_AlphaTested) || (donutGS.material.domain == MaterialDomain_TransmissiveAlphaTested);
    ptShadingData.materialID = donutGS.geometry.materialIndex;
    ptShadingData.mtl = MaterialHeader::make();
    //ptShadingData.mtl.setMaterialType( MaterialType::Standard );
    //ptShadingData.mtl.setAlphaMode( (AlphaMode)( (!alphaTested)?((uint)AlphaMode::Opaque):((uint)AlphaMode::Mask) ) );    // alpha testing handled on our side, Falcor stuff is unused
    //ptShadingData.mtl.setAlphaThreshold( donutGS.material.alphaCutoff );                                                  // alpha testing handled on our side, Falcor stuff is unused
    ptShadingData.mtl.setNestedPriority( min( InteriorList::kMaxNestedPriority, 1 + (uint(donutGS.material.flags) >> MaterialFlags_NestedPriorityShift)) );   // priorities are from (1, ... kMaxNestedPriority) because 0 is used to mark empty slots and remapped to kMaxNestedPriority
    //ptShadingData.mtl.setDoubleSided( donutMaterialDoubleSided );  // removed; all triangles are double sided to avoid breaking path tracing logic in various cases
    ptShadingData.mtl.setThinSurface( donutMaterialThinSurface );
    ptShadingData.mtl.setEmissive( any(donutMaterial.emissiveColor!=0) );
    //ptShadingData.mtl.setIsBasicMaterial( true );
    ptShadingData.mtl.setPSDExclude( (donutGS.material.flags & MaterialFlags_PSDExclude) != 0 );
    ptShadingData.mtl.setPSDDominantDeltaLobeP1( (donutGS.material.flags & MaterialFlags_PSDDominantDeltaLobeP1Mask) >> MaterialFlags_PSDDominantDeltaLobeP1Shift );

    // We flip the shading normal for back-facing hits on double-sided materials, and we currently consider all surfaces to be double-sided.
    // This convention will eventually go away when the material setup code handles it instead.
    if (!ptShadingData.frontFacing)
        ptShadingData.N = -ptShadingData.N;

    // Helper function to adjust the shading normal to reduce black pixels due to back-facing view direction. Note: This breaks the reciprocity of the BSDF!
    // This also reorthonormalizes the frame based on the normal map, which is necessary (see `ptShadingData.N = donutMaterial.shadingNormal;` line above)
    adjustShadingNormal( ptShadingData, ptVertex, true );

    ptShadingData.opacity = donutMaterial.opacity;

    ptShadingData.shadowNoLFadeout = donutMaterial.shadowNoLFadeout;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Now load the actual BSDF! Equivalent to StandardBSDF::setupBSDF
    StandardBSDFData d = StandardBSDFData::make();

    // A.k.a. interiorIoR
    float matIoR = donutMaterial.ior;

    // from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#refraction
    // "This microfacet lobe is exactly the same as the specular lobe except sampled along the line of sight through the surface."
    d.specularTransmission = donutMaterial.transmission * (1 - donutMaterial.metalness);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    d.diffuseTransmission = donutMaterial.diffuseTransmission * (1 - donutMaterial.metalness);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    d.transmission = donutMaterial.baseColor;
    d.scatter = donutMaterial.scatter;
    d.sssMeanFreePath = donutMaterial.sssMeanFreePath;
    d.ssSurfaceAlbedo = donutMaterial.ssSurfaceAlbedo;
    d.modelId = donutMaterial.modelId;
    d.IrisNormal = donutMaterial.irisNormal;
    d.IrisMask = donutMaterial.irisMask;
    d.CausticNormal = donutMaterial.causticNormal;

    /*LobeType*/ uint lobeType = (uint)LobeType::All;

    if (optimizationHints.NoTransmission)
    {
        d.specularTransmission = 0;
        d.diffuseTransmission = 0;
        d.transmission = float3(0,0,0);
        lobeType &= ~(uint)LobeType::Transmission;//~((uint)LobeType::DiffuseReflection | (uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection);
    }
    //if (optimizationHints.OnlyTransmission)
    //{
    //    lobeType &= (uint)LobeType::Transmission; //~(uint)LobeType::Reflection;
    //}
    if (optimizationHints.OnlyDeltaLobes)
    {
        lobeType &= ~(uint)LobeType::NonDelta;
    }

    ptShadingData.mtl.setActiveLobes( lobeType );

    // Sample base color.
    float3 baseColor = donutMaterial.baseColor;

    // OMM Debug evaluates the OMM state at a given triangle + hit BC color codes the result for the corresonding state.
    OpacityMicroMapDebugInfo ommDebug = loadOmmDebugInfo(donutGS, triangleIndex, triangleHit);
#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS
    if (ommDebug.hasOmmAttachment && 
        g_Const.debug.debugViewType == (int)DebugViewType::FirstHitOpacityMicroMapInWorld)
    {
        baseColor = ommDebug.opacityStateDebugColor;
    }
#endif

#if ENABLE_METAL_ROUGH_RECONSTRUCTION == 0
#error we rely on Donut to do the conversion! for more info on how to do it manually search for MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS 
#endif

    // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
    // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
    float f = (matIoR - 1.f) / (matIoR + 1.f);
    float F0 = f * f;

    // G - Roughness; B - Metallic
    d.diffuse = lerp(baseColor, float3(0,0,0), donutMaterial.metalness);
    d.specular = lerp(float3(F0,F0,F0), baseColor, donutMaterial.metalness);
    d.roughness = donutMaterial.roughness;
    d.metallic = donutMaterial.metalness;

    // Assume the default IoR for vacuum on the front-facing side.
    // The renderer may override this for nested dielectrics (see 'handleNestedDielectrics' calling Bridge::updateOutsideIoR)
    ptShadingData.IoR = 1.f;
    d.eta = ptShadingData.frontFacing ? (ptShadingData.IoR / matIoR) : (matIoR / ptShadingData.IoR); 

    StandardBSDF bsdf = StandardBSDF::make();
    bsdf.data = d;


    // Sample the emissive texture.
    // The standard material supports uniform emission over the hemisphere.
    // Note: we only support single sided emissives at the moment; If upgrading, make sure to upgrade NEE codepath as well (i.e. PolymorphicLight.hlsli)
    bsdf.emission = (ptShadingData.frontFacing)?(donutMaterial.emissiveColor):(0);

    // if you think tangent space is broken, test with this (won't make it correctly oriented)
    //ConstructONB( ptShadingData.N, ptShadingData.T, ptShadingData.B );

#if RLSHADER == Eye
    //d.diffuse = d.CausticNormal;
#endif
    
    PathTracer::SurfaceData ret = PathTracer::SurfaceData::make(/*ptVertex, */ptShadingData, bsdf, prevPosW, matIoR);

#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS
    if( debug.IsDebugPixel() && pathVertexIndex==1 && !debug.constants.exploreDeltaTree )
        debug.SetPickedMaterial( donutGS.geometry.materialIndex );
    surfaceDebugViz( optimizationHints, ret, triangleHit, rayDir, rayCone, pathVertexIndex, ommDebug, debug );
#endif
    return ret;
}

void Bridge::updateOutsideIoR(inout PathTracer::SurfaceData surfaceData, float outsideIoR)
{
    surfaceData.shadingData.IoR = outsideIoR;

    ///< Relative index of refraction (incident IoR / transmissive IoR), dependent on whether we're exiting or entering
    surfaceData.bsdf.data.eta = surfaceData.shadingData.frontFacing ? (surfaceData.shadingData.IoR / surfaceData.interiorIoR) : (surfaceData.interiorIoR / surfaceData.shadingData.IoR); 
}

float Bridge::loadIoR(const uint materialID)
{
    if( materialID >= g_Const.materialCount )
        return 1.0;
    else
        return t_MaterialConstants[materialID].ior;
}

HomogeneousVolumeData Bridge::loadHomogeneousVolumeData(const uint materialID)
{
    HomogeneousVolumeData ptVolume;
    ptVolume.sigmaS = float3(0,0,0); 
    ptVolume.sigmaA = float3(0,0,0); 
    ptVolume.g = 0.0;

    if( materialID >= g_Const.materialCount )
        return ptVolume;

    VolumeConstants donutVolume = t_MaterialConstants[materialID].volume;
        
    // these should be precomputed on the C++ side!!
    ptVolume.sigmaS = float3(0,0,0); // no scattering yet
    ptVolume.sigmaA = -log( clamp( donutVolume.attenuationColor, 1e-7, 1 ) ) / max( 1e-30, donutVolume.attenuationDistance.xxx );

    return ptVolume;        
}

// 2.5D motion vectors
float3 Bridge::computeMotionVector( float3 posW, float3 prevPosW )
{
    PlanarViewConstants view = g_Const.view;
    PlanarViewConstants previousView = g_Const.previousView;

    float4 clipPos = mul(float4(posW, 1), view.matWorldToClipNoOffset);
    clipPos.xyz /= clipPos.w;
    float4 prevClipPos = mul(float4(prevPosW, 1), previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return float3(0,0,0);

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); //<- no longer needed, using NoOffset matrices
    motion.z = prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}
// 2.5D motion vectors
float3 Bridge::computeSkyMotionVector( const uint2 pixelPos )
{
    PlanarViewConstants view = g_Const.view;
    PlanarViewConstants previousView = g_Const.previousView;

    float4 clipPos = float4( (pixelPos + 0.5.xx)/g_Const.view.clipToWindowScale+float2(-1,1), 1e-7, 1.0);
    float4 viewPos = mul( clipPos, view.matClipToWorldNoOffset ); viewPos.xyzw /= viewPos.w;
    float4 prevClipPos = mul(viewPos, previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); <- no longer needed, using NoOffset matrices
    motion.z = 0; //prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}

bool AlphaTestImpl(SubInstanceData subInstanceData, uint triangleIndex, float2 rayBarycentrics)
{
    bool alphaTested = (subInstanceData.FlagsAndSERSortKey & SubInstanceData::Flags_AlphaTested) != 0;
    if( !alphaTested ) // note: with correct use of D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE this is unnecessary, but there are cases (such as disabling texture but leaving alpha tested state) in which this isn't handled correctly
        return true;
        
    // have to do all this to figure out UVs!
    float2 texcoord;
    {
        GeometryData geometry = t_GeometryData[NonUniformResourceIndex(subInstanceData.GlobalGeometryIndex)];

        ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
        ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];

        float3 barycentrics;
        barycentrics.yz = rayBarycentrics;
        barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);

        uint3 indices = indexBuffer.Load3(geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);

        float2 vertexTexcoords[3];
        vertexTexcoords[0] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        vertexTexcoords[1] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        vertexTexcoords[2] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
        texcoord = interpolate(vertexTexcoords, barycentrics);
    }
    // sample the alpha (opacity) texture and test vs the threshold
    Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(subInstanceData.AlphaTextureIndex)];
    float opacityValue = diffuseTexture.SampleLevel(s_MaterialSampler, texcoord, 0).a; // <- hard coded to .a channel but we might want a separate alpha only texture, maybe in .g of BC1
    return opacityValue >= subInstanceData.AlphaCutoff;
}

bool Bridge::AlphaTest(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[NonUniformResourceIndex(instanceID + geometryIndex)];

    return AlphaTestImpl(subInstanceData, triangleIndex, rayBarycentrics);
}

bool Bridge::AlphaTestVisibilityRay(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[NonUniformResourceIndex(instanceID + geometryIndex)];

    bool excludeFromNEE = (subInstanceData.FlagsAndSERSortKey & SubInstanceData::Flags_ExcludeFromNEE) != 0;
    if (excludeFromNEE)
        return false;

    return AlphaTestImpl(subInstanceData, triangleIndex, rayBarycentrics);
}

// There's a relatively high cost to this when used in large shaders just due to register allocation required for alphaTest, even if all geometries are opaque.
// Consider simplifying alpha testing - perhaps splitting it up from the main geometry path, load it with fewer indirections or something like that.
bool Bridge::traceVisibilityRay(RayDesc ray, const RayCone rayCone, const int pathVertexIndex, DebugContext debug)
{
#if 0
    #error make sure to enable specialized "visibility miss shader" for this to work
    const uint missShaderIndex = 1; // visibility miss shader
    VisibilityPayload visibilityPayload = VisibilityPayload::make();     // will be set to 1 if miss shader called
    TraceRay(SceneBVH, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, 0, 0, missShaderIndex, ray, visibilityPayload);
    return visibilityPayload.missed != 0;
#else
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            [branch]if (Bridge::AlphaTestVisibilityRay(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, debug
                )
            )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
                // break; <- TODO: revisit - not needed when using RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH?
            }
        }
    }

        
#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    float visible = !(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        ray.TMax = rayQuery.CommittedRayT();    // <- this gets passed via NvMakeHitWithRecordIndex/NvInvokeHitObject as RayTCurrent() or similar in ubershader path

    if( debug.IsDebugPixel() )
        debug.DrawLine(ray.Origin, ray.Origin+ray.Direction*ray.TMax, float4(visible.x, visible.x, 1-visible.x, 0.5), float4(visible.x, 1-visible.x, 1-visible.x, 0.8));
#endif

    return !(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
#endif
}

void Bridge::traceSssProfileRadiusRay(RayDesc ray, inout RayQuery<RAY_FLAG_NONE> rayQuery, inout PackedHitInfo packedHitInfo, DebugContext debug)
{
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // A.k.a. 'Anyhit' shader!
            [branch]if (Bridge::AlphaTest(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, workingContext.debug
                )
            )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        ray.TMax = rayQuery.CommittedRayT();    // <- this gets passed via NvMakeHitWithRecordIndex/NvInvokeHitObject as RayTCurrent() or similar in ubershader path

        TriangleHit triangleHit;
        triangleHit.instanceID      = GeometryInstanceID::make( rayQuery.CommittedInstanceIndex(), rayQuery.CommittedGeometryIndex() );
        triangleHit.primitiveIndex  = rayQuery.CommittedPrimitiveIndex();
        triangleHit.barycentrics    = rayQuery.CommittedTriangleBarycentrics(); // attrib.barycentrics;
        packedHitInfo = triangleHit.pack();
    }
    else
    {
        packedHitInfo = PACKED_HIT_INFO_ZERO; // this invokes miss shader a.k.a. sky!
    }
}

void Bridge::traceScatterRay(const PathState path, inout RayDesc ray, inout RayQuery<RAY_FLAG_NONE> rayQuery, inout PackedHitInfo packedHitInfo, inout uint SERSortKey, DebugContext debug)
{
    ray = path.getScatterRay().toRayDesc();
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // A.k.a. 'Anyhit' shader!
            [branch]if (Bridge::AlphaTest(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, workingContext.debug
                )
            )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        ray.TMax = rayQuery.CommittedRayT();    // <- this gets passed via NvMakeHitWithRecordIndex/NvInvokeHitObject as RayTCurrent() or similar in ubershader path

        TriangleHit triangleHit;
        triangleHit.instanceID      = GeometryInstanceID::make( rayQuery.CommittedInstanceIndex(), rayQuery.CommittedGeometryIndex() );
        triangleHit.primitiveIndex  = rayQuery.CommittedPrimitiveIndex();
        triangleHit.barycentrics    = rayQuery.CommittedTriangleBarycentrics(); // attrib.barycentrics;
        packedHitInfo = triangleHit.pack();

        // per-instance sort key from cpp side - only needed if USE_UBERSHADER_IN_SER used
        SERSortKey = t_SubInstanceData[rayQuery.CommittedInstanceID()+rayQuery.CommittedGeometryIndex()].FlagsAndSERSortKey & 0xFFFF;
    }
    else
    {
        packedHitInfo = PACKED_HIT_INFO_ZERO; // this invokes miss shader a.k.a. sky!
        SERSortKey = 0;
    }
}

void Bridge::StoreSecondarySurfacePositionAndNormal(uint2 pixelCoordinate, float3 worldPos, float3 normal)
{
    const uint encodedNormal = ndirToOctUnorm32(normal);
    u_SecondarySurfacePositionNormal[pixelCoordinate] = float4(worldPos, asfloat(encodedNormal));
}

EnvMap Bridge::CreateEnvMap()
{
    return EnvMap::make( t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams );
}

EnvMapSampler Bridge::CreateEnvMapImportanceSampler()
{
    return EnvMapSampler::make(
        s_EnvironmentMapImportanceSampler,
        t_EnvironmentMapImportanceMap,
        g_Const.envMapImportanceSamplingParams,
        t_EnvironmentMap,
        s_EnvironmentMapSampler,
        g_Const.envMapSceneParams,
        t_PresampledEnvMapBuffer
    );
}

bool Bridge::HasEnvMap()
{
    return g_Const.ptConsts.hasEnvMap;
}

#endif // __PATH_TRACER_BRIDGE_DONUT_HLSLI__