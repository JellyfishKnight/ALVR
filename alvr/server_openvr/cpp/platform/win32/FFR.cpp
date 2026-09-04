#include "FFR.h"

#include "alvr_server/Utils.h"
#include "alvr_server/bindings.h"

#include <algorithm>
#include <cmath>

using Microsoft::WRL::ComPtr;
using namespace d3d_render_utils;

namespace {

struct FoveationVars {
    uint32_t targetEyeWidth;
    uint32_t targetEyeHeight;
    uint32_t optimizedEyeWidth;
    uint32_t optimizedEyeHeight;

    float eyeWidthRatio;
    float eyeHeightRatio;

    float centerSizeX;
    float centerSizeY;
    float centerShiftLeftX;
    float centerShiftLeftY;
    float centerShiftRightX;
    float centerShiftRightY;
    float edgeRatioX;
    float edgeRatioY;
    float padding[2];
};

static_assert(sizeof(FoveationVars) == 64, "FoveationVars must match the HLSL constant buffer");

float AlignCenterShift(float centerShift, float edgeSizeAligned, float edgeRatio) {
    if (!std::isfinite(centerShift) || !std::isfinite(edgeSizeAligned) || !std::isfinite(edgeRatio)
        || edgeSizeAligned <= 0.0f || edgeRatio <= 0.0f) {
        return 0.0f;
    }

    const float alignmentStep = edgeRatio * 2.0f / edgeSizeAligned;
    if (!std::isfinite(alignmentStep) || alignmentStep >= 1.0f) {
        return 0.0f;
    }

    // At exactly +/-1 one peripheral segment has zero width, making the inverse
    // transform singular. Keep one encoder-alignment step on both sides so the
    // encoder and decoder transforms remain finite and invertible.
    // Preserve the legacy operation order to avoid crossing an integer boundary after float
    // rounding.
    const float aligned = std::ceil(centerShift * edgeSizeAligned / (edgeRatio * 2.0f))
        * (edgeRatio * 2.0f) / edgeSizeAligned;
    const float safeMin = -1.0f + alignmentStep;
    const float safeMax = 1.0f - alignmentStep;

    return std::clamp(aligned, safeMin, safeMax);
}

FoveationVars CalculateFoveationVars(
    float centerShiftLeftX, float centerShiftLeftY, float centerShiftRightX, float centerShiftRightY
) {
    float targetEyeWidth = (float)Settings_Instance()->m_renderWidth / 2;
    float targetEyeHeight = (float)Settings_Instance()->m_renderHeight;

    float centerSizeX = (float)Settings_Instance()->m_foveationCenterSizeX;
    float centerSizeY = (float)Settings_Instance()->m_foveationCenterSizeY;
    float edgeRatioX = (float)Settings_Instance()->m_foveationEdgeRatioX;
    float edgeRatioY = (float)Settings_Instance()->m_foveationEdgeRatioY;

    float edgeSizeX = targetEyeWidth - centerSizeX * targetEyeWidth;
    float edgeSizeY = targetEyeHeight - centerSizeY * targetEyeHeight;

    float centerSizeXAligned
        = 1. - ceil(edgeSizeX / (edgeRatioX * 2.)) * (edgeRatioX * 2.) / targetEyeWidth;
    float centerSizeYAligned
        = 1. - ceil(edgeSizeY / (edgeRatioY * 2.)) * (edgeRatioY * 2.) / targetEyeHeight;

    float edgeSizeXAligned = targetEyeWidth - centerSizeXAligned * targetEyeWidth;
    float edgeSizeYAligned = targetEyeHeight - centerSizeYAligned * targetEyeHeight;

    float centerShiftLeftXAligned
        = AlignCenterShift(centerShiftLeftX, edgeSizeXAligned, edgeRatioX);
    float centerShiftLeftYAligned
        = AlignCenterShift(centerShiftLeftY, edgeSizeYAligned, edgeRatioY);
    float centerShiftRightXAligned
        = AlignCenterShift(centerShiftRightX, edgeSizeXAligned, edgeRatioX);
    float centerShiftRightYAligned
        = AlignCenterShift(centerShiftRightY, edgeSizeYAligned, edgeRatioY);

    float foveationScaleX = (centerSizeXAligned + (1. - centerSizeXAligned) / edgeRatioX);
    float foveationScaleY = (centerSizeYAligned + (1. - centerSizeYAligned) / edgeRatioY);

    float optimizedEyeWidth = foveationScaleX * targetEyeWidth;
    float optimizedEyeHeight = foveationScaleY * targetEyeHeight;

    // round the frame dimensions to a number of pixel multiple of 32 for the encoder
    auto optimizedEyeWidthAligned = (uint32_t)ceil(optimizedEyeWidth / 32.f) * 32;
    auto optimizedEyeHeightAligned = (uint32_t)ceil(optimizedEyeHeight / 32.f) * 32;

    float eyeWidthRatioAligned = optimizedEyeWidth / optimizedEyeWidthAligned;
    float eyeHeightRatioAligned = optimizedEyeHeight / optimizedEyeHeightAligned;

    return { (uint32_t)targetEyeWidth,
             (uint32_t)targetEyeHeight,
             optimizedEyeWidthAligned,
             optimizedEyeHeightAligned,
             eyeWidthRatioAligned,
             eyeHeightRatioAligned,
             centerSizeXAligned,
             centerSizeYAligned,
             centerShiftLeftXAligned,
             centerShiftLeftYAligned,
             centerShiftRightXAligned,
             centerShiftRightYAligned,
             edgeRatioX,
             edgeRatioY,
             { 0.0f, 0.0f } };
}
}

void FFR::GetOptimizedResolution(uint32_t* width, uint32_t* height) {
    auto fovVars = CalculateFoveationVars(
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY,
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY
    );
    *width = fovVars.optimizedEyeWidth * 2;
    *height = fovVars.optimizedEyeHeight;
}

FFR::FFR(ID3D11Device* device)
    : mDevice(device) { }

void FFR::Initialize(ID3D11Texture2D* compositionTexture) {
    auto fovVars = CalculateFoveationVars(
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY,
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY
    );
    mFoveatedRenderingBuffer = CreateBuffer(mDevice.Get(), fovVars, D3D11_USAGE_DEFAULT);
    mDevice->GetImmediateContext(&mImmediateContext);

    std::vector<uint8_t> quadShaderCSO(
        QUAD_SHADER_CSO_PTR, QUAD_SHADER_CSO_PTR + QUAD_SHADER_CSO_LEN
    );
    mQuadVertexShader = CreateVertexShader(mDevice.Get(), quadShaderCSO);

    mOptimizedTexture = CreateTexture(
        mDevice.Get(),
        fovVars.optimizedEyeWidth * 2,
        fovVars.optimizedEyeHeight,
        Settings_Instance()->m_enableHdr ? DXGI_FORMAT_R16G16B16A16_FLOAT
                                         : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
    );

    if (Settings_Instance()->m_enableFoveatedEncoding) {
        std::vector<uint8_t> compressAxisAlignedShaderCSO(
            COMPRESS_AXIS_ALIGNED_CSO_PTR,
            COMPRESS_AXIS_ALIGNED_CSO_PTR + COMPRESS_AXIS_ALIGNED_CSO_LEN
        );
        auto compressAxisAlignedPipeline = RenderPipeline(mDevice.Get());
        compressAxisAlignedPipeline.Initialize(
            { compositionTexture },
            mQuadVertexShader.Get(),
            compressAxisAlignedShaderCSO,
            mOptimizedTexture.Get(),
            mFoveatedRenderingBuffer.Get()
        );

        mPipelines.push_back(compressAxisAlignedPipeline);
    } else {
        mOptimizedTexture = compositionTexture;
    }
}

void FFR::Render(uint64_t targetTimestampNs) {
    auto fovVars = CalculateFoveationVars(
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY,
        Settings_Instance()->m_foveationCenterShiftX,
        Settings_Instance()->m_foveationCenterShiftY
    );
    UpdateBuffer(mImmediateContext.Get(), mFoveatedRenderingBuffer.Get(), &fovVars);

    if (Settings_Instance()->m_enableFoveationCenterMetadata) {
        // Publish the exact encoder-aligned values used for this frame.
        SetEncoderFoveationCenters(
            targetTimestampNs,
            fovVars.centerShiftLeftX,
            fovVars.centerShiftLeftY,
            fovVars.centerShiftRightX,
            fovVars.centerShiftRightY
        );
    }

    for (auto& p : mPipelines) {
        p.Render();
    }
}

ID3D11Texture2D* FFR::GetOutputTexture() { return mOptimizedTexture.Get(); }
