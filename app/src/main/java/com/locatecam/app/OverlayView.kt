package com.locatecam.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var srcW = 1
    private var srcH = 1
    private var labels: ((Int) -> String)? = null

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        alpha = 60
    }
    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 42f
        isFakeBoldText = true
    }

    private val palette = intArrayOf(
        0xFF4FC3F7.toInt(), 0xFFFFB74D.toInt(), 0xFF81C784.toInt(), 0xFFE57373.toInt(),
        0xFFBA68C8.toInt(), 0xFFFFD54F.toInt(), 0xFF4DB6AC.toInt(), 0xFFF06292.toInt(),
        0xFF9575CD.toInt(), 0xFFAED581.toInt()
    )

    fun update(detections: List<Detection>, srcW: Int, srcH: Int, labels: (Int) -> String) {
        this.detections = detections
        this.srcW = maxOf(1, srcW)
        this.srcH = maxOf(1, srcH)
        this.labels = labels
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return
        val labels = this.labels ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        val scale = maxOf(vw / srcW, vh / srcH)
        val dx = (vw - srcW * scale) / 2f
        val dy = (vh - srcH * scale) / 2f

        val rect = RectF()
        for (det in detections) {
            val color = palette[Math.floorMod(det.labelIndex, palette.size)]
            boxPaint.color = color
            fillPaint.color = color
            textBgPaint.color = color

            rect.set(
                det.box.left * scale + dx,
                det.box.top * scale + dy,
                det.box.right * scale + dx,
                det.box.bottom * scale + dy
            )
            canvas.drawRect(rect, fillPaint)
            canvas.drawRect(rect, boxPaint)

            val label = labels(det.labelIndex)
            val text = "$label ${(det.score * 100).toInt()}%"
            val tw = textPaint.measureText(text)
            var tx = rect.left
            var ty = rect.top - 48f
            if (ty < 0) ty = 0f
            if (tx + tw > vw) tx = vw - tw
            if (tx < 0) tx = 0f
            canvas.drawRect(tx, ty, tx + tw, ty + 48f, textBgPaint)
            canvas.drawText(text, tx, ty + 38f, textPaint)
        }
    }
}
