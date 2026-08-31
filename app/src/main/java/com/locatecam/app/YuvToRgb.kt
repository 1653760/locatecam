package com.locatecam.app

import androidx.camera.core.ImageProxy

object YuvToRgb {

    const val OUT = 640

    fun convert(image: ImageProxy, out: IntArray): IntArray {
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

        val ow = OUT
        val oh = OUT

        for (oy in 0 until oh) {
            val ny = oy.toFloat() / (oh - 1)
            for (ox in 0 until ow) {
                val nx = ox.toFloat() / (ow - 1)
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
        return out
    }

    fun uprightSize(image: ImageProxy): Pair<Int, Int> {
        return if (image.imageInfo.rotationDegrees == 90 || image.imageInfo.rotationDegrees == 270) {
            image.height to image.width
        } else {
            image.width to image.height
        }
    }
}
