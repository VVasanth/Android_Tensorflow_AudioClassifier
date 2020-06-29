package com.ml.quaterion.noiseClassification


class Recognition {


    private var id : String? = null
    private var title : String? = null
    private var confidence : Float

    constructor(id: String, title: String, confidence: Float){

        this.id = id
        this.title = title
        this.confidence = confidence
    }

    fun getId(): String? {
        return id
    }

    fun getTitle(): String? {
        return title
    }

    fun getConfidence(): Float {
        return confidence
    }


}