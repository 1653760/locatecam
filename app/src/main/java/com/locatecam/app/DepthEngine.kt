package com.locatecam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer

/**
 * Depth Anything V2 Metric (Indoor, ViT-S) distance estimation.
 * Input: raw RGB floats [0,255], [1,3,518,518] (full frame, aspect-stretched).
 * Output: metric depth map in meters; we take the median over the target box.
 */
class DepthEngine(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession
    private var inputName: String
    private val depthBuf = FloatArray(3 * IN * IN)
    private val flatDepth = FloatArray(IN * IN)
    private val sampleScratch = FloatArray(IN * IN)

    init {
        val bytes = context.assets.open("depth.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputInfo.keys.first()
        Log.i(TAG, "depth engine ready, input=$inputName")
    }

    /**
     * Estimate distance (meters) from the camera to the target box.
     * frame: upright full-frame RGB pixels (ARGB ints), w×h
     * box: target box in the same frame coordinate space
     * Returns median depth over the central region of the box, or -1 on failure.
     */
    fun estimate(frame: IntArray, w: Int, h: Int, box: RectF): Float {
        try {
            fillTensor(frame, w, h)
            val shape = longArrayOf(1, 3, IN.toLong(), IN.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(depthBuf), shape).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { out ->
                    flatten(out[0].value)
                }
            }
            // map box (frame space) into depth-map space (IN×IN, aspect-stretched)
            val sx = IN.toFloat() / w
            val sy = IN.toFloat() / h
            val cx0 = (box.left + 0.2f * box.width()) * sx
            val cx1 = (box.right - 0.2f * box.width()) * sx
            val cy0 = (box.top + 0.2f * box.height()) * sy
            val cy1 = (box.bottom - 0.2f * box.height()) * sy
            val x0 = cx0.toInt().coerceIn(0, IN - 1)
            val x1 = cx1.toInt().coerceIn(x0 + 1, IN)
            val y0 = cy0.toInt().coerceIn(0, IN - 1)
            val y1 = cy1.toInt().coerceIn(y0 + 1, IN)
            var n = 0
            for (y in y0 until y1 step 2) {
                val row = y * IN
                for (x in x0 until x1 step 2) {
                    sampleScratch[n++] = flatDepth[row + x]
                }
            }
            if (n == 0) return -1f
            java.util.Arrays.sort(sampleScratch, 0, n)
            // 30th percentile: bias toward the nearest surface, robust to background
            // bleed at box edges (box tends to be slightly larger than the object)
            val med = sampleScratch[(n * 3) / 10]
            return if (med in 0.05f..40f) med else -1f
        } catch (t: Throwable) {
            Log.w(TAG, "depth estimate failed: ${t.message}")
            return -1f
        }
    }

    /** Nearest-neighbour resample full frame -> IN×IN, channels as raw 0-255 floats. */
    private fun fillTensor(frame: IntArray, w: Int, h: Int) {
        val n = IN * IN
        for (dy in 0 until IN) {
            val sy = dy * h / IN
            val rowOut = dy * IN
            val rowIn = sy * w
            for (dx in 0 until IN) {
                val sx = dx * w / IN
                val p = frame[rowIn + sx]
                val idx = rowOut + dx
                depthBuf[idx] = ((p shr 16) and 0xFF).toFloat()
                depthBuf[n + idx] = ((p shr 8) and 0xFF).toFloat()
                depthBuf[2 * n + idx] = (p and 0xFF).toFloat()
            }
        }
    }

    /** Flatten nested ONNX output container into flatDepth (IN*IN floats). */
    private fun flatten(value: Any) {
        sink.clear()
        collectRows(value, sink)
        // sink should end up as IN rows of IN floats
        var k = 0
        for (row in sink) {
            for (v in row) {
                if (k < flatDepth.size) flatDepth[k++] = v
            }
        }
    }

    private val sink = ArrayList<FloatArray>(IN * 2)

    private fun collectRows(node: Any, out: MutableList<FloatArray>) {
        if (node is FloatArray) {
            out.add(node)
        } else if (node is Array<*>) {
            for (child in node) if (child != null) collectRows(child, out)
        } else if (node is java.util.List<*>) {
            for (child in node) if (child != null) collectRows(child as Any, out)
        }
    }

    fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "DepthEngine"
        private const val IN = 518
    }
}
