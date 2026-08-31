package com.locatecam.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class VocabEntry(val en: String, val zh: List<String>)

class Vocab(context: Context) {

    val entries: List<VocabEntry>

    init {
        val text = context.assets.open("class_index.json").bufferedReader().use { it.readText() }
        val arr = JSONArray(text)
        val list = ArrayList<VocabEntry>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val zhArr = o.optJSONArray("zh")
            val zh = ArrayList<String>()
            if (zhArr != null) for (j in 0 until zhArr.length()) zh.add(zhArr.getString(j))
            list.add(VocabEntry(o.getString("en"), zh))
        }
        entries = list
    }

    fun indexOf(term: String): Int {
        val q = term.trim().lowercase()
        if (q.isEmpty()) return -1
        entries.forEachIndexed { i, e ->
            if (e.en.lowercase() == q) return i
        }
        entries.forEachIndexed { i, e ->
            if (e.zh.any { it == term.trim() }) return i
        }
        entries.forEachIndexed { i, e ->
            if (e.zh.any { it.contains(term.trim()) || term.trim().contains(it) && it.length >= 2 }) return i
        }
        return -1
    }

    fun display(i: Int): String = entries.getOrNull(i)?.zh?.firstOrNull() ?: entries.getOrNull(i)?.en ?: "?"
}
