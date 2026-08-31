package com.locatecam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer
import java.util.PriorityQueue

data class Detection(val labelIndex: Int, val score: Float, val box: RectF)

class Timing(val preMs: Long, val inferMs: Long, val postMs: Long)

class DetectionEngine(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val inputSize = 640
    private val numAnchors = 8400
    private val confidenceThreshold = 0.28f
    private val iouThreshold = 0.45f

    private val floatBuf = FloatBuffer.wrap(FloatArray(1 * 3 * inputSize * inputSize))
    private val scoreBuf = FloatArray(numAnchors)
    private var boxRows = Array(4) { FloatArray(numAnchors) }
    private val detectedClasses = ArrayList<Int>()

    var selectedIndices: List<Int> = emptyList()
    var vocab: Vocab? = null

    init {
        val bytes = context.assets.open("yolo_world.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            try {
                addNnapi()
                Log.d(TAG, "NNAPI enabled")
            } catch (e: Throwable) {
                Log.w(TAG, "NNAPI unavailable, CPU fallback: ${e.message}")
                setIntraOpNumThreads(4)
            }
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        Log.i(TAG, "model loaded, inputs=${session!!.inputInfo.keys}, output=${session!!.outputInfo.keys}")
    }

    fun warmup() {
        detect(IntArray(inputSize * inputSize) { -0x1000000 }, inputSize, inputSize)
    }

    fun detect(rgb: IntArray, srcW: Int, srcH: Int): Pair<List<Detection>, Timing> {
        val t0 = System.nanoTime()
        fillTensor(rgb)
        val t1 = System.nanoTime()

        val session = this.session ?: return emptyList<Detection>() to Timing(0, 0, 0)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        OnnxTensor.createTensor(env, floatBuf.duplicate().rewind() as FloatBuffer, shape).use { tensor ->
            val inputs = mapOf("images" to tensor)
            session.run(inputs).use { outputs ->
                val t2 = System.nanoTime()
                val out = outputs[0].value
                val dets = parseOutput(out, srcW, srcH)
                val t3 = System.nanoTime()
                return dets to Timing((t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000, (t3 - t2) / 1_000_000)
            }
        }
    }

    private fun fillTensor(rgb: IntArray) {
        val f = floatBuf.array()
        val n = inputSize * inputSize
        var i = 0
        for (y in 0 until inputSize) {
            val row = y * inputSize
            for (x in 0 until inputSize) {
                val p = rgb[row + x]
                f[i] = ((p shr 16) and 0xFF) / 255f
                f[n + i] = ((p shr 8) and 0xFF) / 255f
                f[2 * n + i] = (p and 0xFF) / 255f
                i++
            }
        }
    }

    private fun parseOutput(value: Any, srcW: Int, srcH: Int): List<Detection> {
        val sel = selectedIndices
        if (sel.isEmpty()) return emptyList()
        val ncTotal: Int
        val fb: FloatBuffer
        val rows: Array<FloatArray>
        if (value is FloatBuffer) {
            fb = value
            ncTotal = (fb.limit() / numAnchors) - 4
            rows = readRowsFb(fb, ncTotal)
        } else if (value is Array<*> && value.size == 1 && value[0] is Array<*>) {
            @Suppress("UNCHECKED_CAST")
            val m = value[0] as Array<FloatArray>
            ncTotal = m.size - 4
            rows = readRowsArr(m, ncTotal)
        } else {
            return emptyList()
        }

        val scaleX = srcW.toFloat() / inputSize
        val scaleY = srcH.toFloat() / inputSize
        val candidates = ArrayList<Detection>()

        for (a in 0 until numAnchors) {
            var bestScore = 0f
            var bestCi = -1
            for (k in sel.indices) {
                val s = rows[4 + sel[k]][a]
                if (s > bestScore) {
                    bestScore = s
                    bestCi = sel[k]
                }
            }
            if (bestScore < confidenceThreshold || bestCi < 0) continue
            val cx = rows[0][a]
            val cy = rows[1][a]
            val w = rows[2][a]
            val h = rows[3][a]
            val x1 = (cx - w / 2f) * scaleX
            val y1 = (cy - h / 2f) * scaleY
            val x2 = (cx + w / 2f) * scaleX
            val y2 = (cy + h / 2f) * scaleY
            candidates.add(
                Detection(
                    bestCi,
                    bestScore,
                    RectF(
                        x1.coerceIn(0f, srcW.toFloat()),
                        y1.coerceIn(0f, srcH.toFloat()),
                        x2.coerceIn(0f, srcW.toFloat()),
                        y2.coerceIn(0f, srcH.toFloat())
                    )
                )
            )
        }
        return nms(candidates)
    }

    private fun readRowsFb(fb: FloatBuffer, ncTotal: Int): Array<FloatArray> {
        val rows = Array(4 + ncTotal) { FloatArray(numAnchors) }
        for (i in rows.indices) {
            fb.position(i * numAnchors)
            fb.get(rows[i], 0, numAnchors)
        }
        return rows
    }

    private fun readRowsArr(m: Array<FloatArray>, ncTotal: Int): Array<FloatArray> {
        val rows = Array(4 + ncTotal) { FloatArray(numAnchors) }
        for (i in 0 until 4) {
            System.arraycopy(m[i], 0, rows[i], 0, numAnchors)
        }
        for (k in selectedIndices) {
            System.arraycopy(m[4 + k], 0, rows[4 + k], 0, numAnchors)
        }
        return rows
    }

    private fun nms(list: List<Detection>): List<Detection> {
        val byClass = list.groupBy { it.labelIndex }
        val out = ArrayList<Detection>()
        for ((_, dets) in byClass) {
            val sorted = dets.sortedByDescending { it.score }
            val keep = BooleanArray(sorted.size) { true }
            for (i in sorted.indices) {
                if (!keep[i]) continue
                for (j in i + 1 until sorted.size) {
                    if (!keep[j]) continue
                    if (iou(sorted[i].box, sorted[j].box) > iouThreshold) keep[j] = false
                }
            }
            for (i in sorted.indices) if (keep[i]) out.add(sorted[i])
        }
        return out
    }

    private fun iou(a: RectF, b: RectF): Float {
        val x1 = maxOf(a.left, b.left)
        val y1 = maxOf(a.top, b.top)
        val x2 = minOf(a.right, b.right)
        val y2 = minOf(a.bottom, b.bottom)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union > 0) inter / union else 0f
    }

    fun close() {
        session?.close()
    }

    companion object {
        private const val TAG = "DetectionEngine"
    }
}
