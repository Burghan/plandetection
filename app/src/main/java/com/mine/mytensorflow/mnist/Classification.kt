package com.mine.mytensorflow.mnist

class Classification(val title: String, val confidence: Float) {

    override fun toString(): String {
        return title + " " + String.format("(%.1f%%) ", confidence * 100.0f)
    }
}
