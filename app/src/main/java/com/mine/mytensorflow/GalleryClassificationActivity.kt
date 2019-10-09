package com.mine.mytensorflow

import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.text.format.DateUtils
import android.util.DisplayMetrics
import android.util.Log
import android.widget.Toast
import com.mine.mytensorflow.inception.InceptionClassifier
import com.mine.mytensorflow.inception.InceptionConfig

import com.mine.mytensorflow.inception_restnet.*
import com.mine.mytensorflow.restnet.*
import com.mine.mytensorflow.mnastnet.*

import com.mine.mytensorflow.mnist.ImageUtils
import com.mine.mytensorflow.mobilenet.MobilenetClassifier
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig
import kotlinx.android.synthetic.main.activity_gallery_classification.*
import kotlinx.android.synthetic.main.activity_gallery_classification.options
import java.io.IOException

class GalleryClassificationActivity : AppCompatActivity() {

    private val SELECT_PICTURE = 100

    private var filepath: Uri? = null
    private var original_image: Bitmap? = null
    private var mobilenetClassifier: MobilenetClassifier? = null
    private var inceptionClassifier: InceptionClassifier? = null


    private var inceptionrestnetClassifier: InceptionRestNetClassifier? = null
    private var restnetClassifier:RestNetClassifier? = null
    private var mnastClassifier:MnastNetClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gallery_classification)

        val filepath = intent.getStringExtra("filepath")
        displaySelectedImage(filepath)
        loadMobilenetClassifier()
        loadInceptionClassifier()

        loadMnastClassifier()
        loadInceptionRestnetClassifier()
        loadRestnetClassifier()

    }

    override fun onBackPressed() {
        super.onBackPressed()
        finish()
        overridePendingTransition(0, 0)
    }


    private fun displaySelectedImage(data: String?) {
        Log.d("Gallery", "filepath:" + data)
        data?.let {
            try {
                filepath = Uri.parse(it)
                original_image = MediaStore.Images.Media.getBitmap(contentResolver, filepath)
                imageView.setImageBitmap(original_image)
                btnPredict.setOnClickListener {
                    makePrediction()
                }

            } catch (e: IOException) {
                Log.d("Gallery activity error", "Error : $e")
            }
        } ?: run {
            finish()
        }
    }

    private fun makePrediction() {
        val squareBitmap = ThumbnailUtils.extractThumbnail(original_image, getScreenWidth(), getScreenWidth())
        if (options.selectedItem == "Mobilenet") {
            val startTime = System.currentTimeMillis()
            val preprocessedImage = ImageUtils.prepareImageForClassificationMobilenet(squareBitmap)
            mobilenetClassifier?.let {
                val recognitions = it.recognizeImage(preprocessedImage)
                val endTime = System.currentTimeMillis()
                val difference = endTime - startTime
                val differenceInSeconds = difference / DateUtils.SECOND_IN_MILLIS
                val timeFormatted = DateUtils.formatElapsedTime(differenceInSeconds)
                var hasil = "hasil : \n"
                recognitions.forEach {
                    hasil = hasil + it.title + " confident \n" + it.confidence + "\n"
                }

                val output = recognitions.toString() + "waktu : " + difference
                tvResult.text = output
            }
        }else if (options.selectedItem == "Inception RestNet") {

            val startTime = System.currentTimeMillis()
            val preprocessedImage = ImageUtils.prepareImageForClassificationMobilenet(squareBitmap)
            inceptionrestnetClassifier?.let {
                val recognitions = it.recognizeImage(preprocessedImage)
                val endTime = System.currentTimeMillis()
                val difference = endTime - startTime
                val differenceInSeconds = difference / DateUtils.SECOND_IN_MILLIS
                val timeFormatted = DateUtils.formatElapsedTime(differenceInSeconds)
                var hasil = "hasil : \n"
                recognitions.forEach {
                    hasil = hasil + it.title + " confident \n" + it.confidence + "\n"
                }

                val output = recognitions.toString() + "waktu : " + difference
                tvResult.text = output
            }


        }else if (options.selectedItem == "RestNet") {

            val startTime = System.currentTimeMillis()
            val preprocessedImage = ImageUtils.prepareImageForClassificationMobilenet(squareBitmap)
            restnetClassifier?.let {
                val recognitions = it.recognizeImage(preprocessedImage)
                val endTime = System.currentTimeMillis()
                val difference = endTime - startTime
                val differenceInSeconds = difference / DateUtils.SECOND_IN_MILLIS
                val timeFormatted = DateUtils.formatElapsedTime(differenceInSeconds)
                var hasil = "hasil : \n"
                recognitions.forEach {
                    hasil = hasil + it.title + " confident \n" + it.confidence + "\n"
                }

                val output = recognitions.toString() + "waktu : " + difference
                tvResult.text = output
            }

        }else if (options.selectedItem == "MnasNet") {

            val startTime = System.currentTimeMillis()
            val preprocessedImage = ImageUtils.prepareImageForClassificationMobilenet(squareBitmap)
            mnastClassifier?.let {
                val recognitions = it.recognizeImage(preprocessedImage)
                val endTime = System.currentTimeMillis()
                val difference = endTime - startTime
                val differenceInSeconds = difference / DateUtils.SECOND_IN_MILLIS
                val timeFormatted = DateUtils.formatElapsedTime(differenceInSeconds)
                var hasil = "hasil : \n"
                recognitions.forEach {
                    hasil = hasil + it.title + " confident \n" + it.confidence + "\n"
                }

                val output = recognitions.toString() + "waktu : " + difference
                tvResult.text = output
            }

        } else {
            val startTime = System.currentTimeMillis()
            val preprocessedImage = ImageUtils.prepareImageForClassificationInception(squareBitmap)
            inceptionClassifier?.let {
                val recognitions = it.recognizeImage(preprocessedImage)
                val endTime = System.currentTimeMillis()
                val difference = endTime - startTime
                val differenceInSeconds = difference / DateUtils.SECOND_IN_MILLIS
                val timeFormatted = DateUtils.formatElapsedTime(differenceInSeconds)
                var hasil = "hasil : \n"
                recognitions.forEach {
                    hasil = hasil + it.title + " confident \n" + it.confidence + "\n"
                }

                val output = recognitions.toString() + "waktu : " + difference
                tvResult.text = output
            }

        }
    }




    private fun loadRestnetClassifier() {
        try {
            restnetClassifier = RestNetClassifier.classifier(assets, RestNetConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "MNIST model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }

    }

    private fun loadInceptionRestnetClassifier() {
        try {
            inceptionrestnetClassifier = InceptionRestNetClassifier.classifier(assets, InceptionRestNetConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "MNIST model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }

    }

    private fun loadMnastClassifier() {
        try {
            mnastClassifier = MnastNetClassifier.classifier(assets, MnasNetConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "MNIST model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }

    }
    private fun loadMobilenetClassifier() {
        try {
            mobilenetClassifier = MobilenetClassifier.classifier(assets, MobilenetModelConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "mobilenet model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT)
                .show()
            e.printStackTrace()
        }

    }

    private fun loadInceptionClassifier() {
        try {
            inceptionClassifier = InceptionClassifier.classifier(assets, InceptionConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "Inception model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT)
                .show()
            e.printStackTrace()
        }

    }

    private fun getScreenWidth(): Int {
        val displayMetrics = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(displayMetrics)
        return displayMetrics.widthPixels
    }
}
