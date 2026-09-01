package com.locatecam.app

import androidx.camera.core.ImageProxy

object YuvToRgb {

    const val OUT = 640

    data class FrameInfo(val w: Int, val h: Int, val cropSize: Int, val offX: Int, val offY: Int)

    fun frameInfo(image: ImageProxy): FrameInfo {
        val rot = image.imageInfo.rotationDegrees
        val w2: Int
        val h2: Int
        if (rot == 90 || rot == 270) {
            w2 = image.height
            h2 = image.width
        } else {
            w2 = image.width
            h2 = image.height
        }
        val crop = minOf(w2, h2)
        return FrameInfo(w2, h2, crop, (w2 - crop) / 2, (h2 - crop) / 2)
    }

    fun convert(image: ImageProxy, out: IntArray): FrameInfo {
        val info = frameInfo(image)
        val w = image.width
        val h = image.height
        val rotation = image.imageInfo.rotationDegrees
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yRow = yPlane.rowStride
        val uRow = uPlane.rowStride
        val vRow = vPlane.rowStride
        val yPix = maxOf(1, yPlane.pixelStride)
        val uPix = maxOf(1, uPlane.pixelStride)
        val vPix = maxOf(1, vPlane.pixelStride)

        val W2 = info.w
        val H2 = info.h
        val ow = OUT
        val oh = OUT

        for (oy in 0 until oh) {
            val cy = oy.toFloat() / (oh - 1)
            for (ox in 0 until ow) {
                val cx = ox.toFloat() / (ow - 1)
                val ux = info.offX + cx * (info.cropSize - 1)
                val uy = info.offY + cy * (info.cropSize - 1)
                val nx = ux / (W2 - 1)
                val ny = uy / (H2 - 1)
                val sx: Float
                val sy: Float
                when (rotation) {
                    90 -> {
                        sx = ny * (w - 1)
                        sy = (1f - nx) * (h - 1)
                    }
                    180 -> {
                        sx = (1f - nx) * (w - 1)
                        sy = (1f - ny) * (h - 1)
                    }
                    270 -> {
                        sx = (1f - ny) * (w - 1)
                        sy = nx * (h - 1)
                    }
                    else -> {
                        sx = nx * (w - 1)
                        sy = ny * (h - 1)
                    }
                }
                val ix = sx.toInt()
                val iy = sy.toInt()

                val y0 = yBuf.get(iy * yRow + ix * yPix).toInt() and 0xFF
                val u0 = uBuf.get((iy shr 1) * uRow + (ix shr 1) * uPix).toInt() - 128
                val v0 = vBuf.get((iy shr 1) * vRow + (ix shr 1) * vPix).toInt() - 128

                var r = y0 + 1.402f * v0
                var g = y0 - 0.344136f * u0 - 0.714136f * v0
                var b = y0 + 1.772f * u0
                if (r < 0f) r = 0f else if (r > 255f) r = 255f
                if (g < 0f) g = 0f else if (g > 255f) g = 255f
                if (b < 0f) b = 0f else if (b > 255f) b = 255f

                out[oy * ow + ox] = -0x1000000 or
                        (r.toInt() shl 16) or
                        (g.toInt() shl 8) or
                        b.toInt()
            }
        }
        return info
    }

    fun convertFull(image: ImageProxy, out: IntArray) {
        val info = frameInfo(image)
        val w = image.width
        val h = image.height
        val rotation = image.imageInfo.rotationDegrees
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yRow = yPlane.rowStride
        val uRow = uPlane.rowStride
        val vRow = vPlane.rowStride
        val yPix = maxOf(1, yPlane.pixelStride)
        val uPix = maxOf(1, uPlane.pixelStride)
        val vPix = maxOf(1, vPlane.pixelStride)

        val W2 = info.w
        val H2 = info.h

        for (oy in 0 until H2) {
            val ny = oy.toFloat() / (H2 - 1)
            for (ox in 0 until W2) {
                val nx = ox.toFloat() / (W2 - 1)
                val sx: Float
                val sy: Float
                when (rotation) {
                    90 -> {
                        sx = ny * (w - 1)
                        sy = (1f - nx) * (h - 1)
                    }
                    180 -> {
                        sx = (1f - nx) * (w - 1)
                        sy = (1f - ny) * (h - 1)
                    }
                    270 -> {
                        sx = (1f - ny) * (w - 1)
                        sy = nx * (h - 1)
                    }
                    else -> {
                        sx = nx * (w - 1)
                        sy = ny * (h - 1)
                    }
                }
                val ix = sx.toInt()
                val iy = sy.toInt()

                val y0 = yBuf.get(iy * yRow + ix * yPix).toInt() and 0xFF
                val u0 = uBuf.get((iy shr 1) * uRow + (ix shr 1) * uPix).toInt() - 128
                val v0 = vBuf.get((iy shr 1) * vRow + (ix shr 1) * vPix).toInt() - 128

                var r = y0 + 1.402f * v0
                var g = y0 - 0.344136f * u0 - 0.714136f * v0
                var b = y0 + 1.772f * u0
                if (r < 0f) r = 0f else if (r > 255f) r = 255f
                if (g < 0f) g = 0f else if (g > 255f) g = 255f
                if (b < 0f) b = 0f else if (b > 255f) b = 255f

                out[oy * W2 + ox] = -0x1000000 or
                        (r.toInt() shl 16) or
                        (g.toInt() shl 8) or
                        b.toInt()
            }
        }
    }
}
