package com.mine.mytensorflow

import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.support.design.widget.BottomNavigationView
import android.app.Activity
import android.graphics.BitmapFactory
import android.media.ThumbnailUtils
import android.util.DisplayMetrics
import android.widget.Toast
import com.mine.mytensorflow.mnist.ImageUtils
import com.mine.mytensorflow.inception.*

import com.mine.mytensorflow.inception_restnet.*
import com.mine.mytensorflow.restnet.*
import com.mine.mytensorflow.mnastnet.*


import com.mine.mytensorflow.mnist.MnistClassifier
import com.mine.mytensorflow.mnist.MnistModelConfig
import com.mine.mytensorflow.mobilenet.MobilenetClassifier
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig
import kotlinx.android.synthetic.main.activity_classification.*
import java.io.IOException
import java.util.*
import android.text.format.DateUtils


import android.os.Environment
import android.graphics.Bitmap

import android.content.pm.PackageManager
import android.Manifest

import android.support.v4.app.ActivityCompat
import com.mine.mytensorflow.inception.InceptionConfig

import java.io.File
import java.io.FileOutputStream


class ClassificationActivity : AppCompatActivity() {

    private var mnistClassifier: MnistClassifier? = null
    private var mobilenetClassifier:MobilenetClassifier? = null
    private var inceptionClassifier:InceptionClassifier? = null

    private var inceptionrestnetClassifier: InceptionRestNetClassifier? = null
    private var restnetClassifier:RestNetClassifier? = null
    private var mnastClassifier:MnastNetClassifier? = null



    internal var imagePath: String? = ""

    val REQUEST_PERM_WRITE_STORAGE = 102
    private val onNavigationItemSelectedListener = BottomNavigationView.OnNavigationItemSelectedListener { item ->

        when (item.itemId) {
            R.id.navigation_home -> {
                val intent = Intent()
                intent.putExtra("pos", 0)
                setResult(Activity.RESULT_OK, intent)
                finish()
                overridePendingTransition(0, 0)
            }
            R.id.navigation_classification -> {
                return@OnNavigationItemSelectedListener false
            }
            R.id.navigation_about -> {
                val intent = Intent()
                intent.putExtra("pos", 2)
                setResult(Activity.RESULT_OK, intent)
                finish()
                overridePendingTransition(0, 0)
            }
        }
        false
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

       // checkPermissions()

        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this@ClassificationActivity,
                arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), REQUEST_PERM_WRITE_STORAGE)

        } else {
            setContentView(R.layout.activity_classification)


            loadMnistClassifier()
            loadMnastClassifier()
            loadInceptionRestnetClassifier()
            loadRestnetClassifier()
            loadMobilenetClassifier()
            loadInceptionClassifier()
            val navView: BottomNavigationView = findViewById(R.id.nav_view)
            navView.selectedItemId = R.id.navigation_classification
            navView.setOnNavigationItemSelectedListener(onNavigationItemSelectedListener)
            btnTakePhoto.setOnClickListener { onTakePhoto() }

        }
    }



    override fun onStart() {
        super.onStart()
        vCamera.onStart()
    }

    override fun onResume() {
        super.onResume()
        vCamera.onResume()
    }

    override fun onPause() {
        vCamera.onPause()
        super.onPause()
    }

    override fun onStop() {
        vCamera.onStop()
        super.onStop()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        vCamera.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    fun onTakePhoto() {
        vCamera.captureImage { cameraKitView, picture -> onImageCaptured(picture) }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        finish()
        overridePendingTransition(0, 0)
    }

    private fun loadMnistClassifier() {
        try {
            mnistClassifier = MnistClassifier.classifier(assets, MnistModelConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "MNIST model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
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
            Toast.makeText(this, "mobilenet model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }

    }

    private fun loadInceptionClassifier() {
        try {
            inceptionClassifier = InceptionClassifier.classifier(assets, InceptionConfig.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(this, "Inception model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }

    }

    private fun onImageCaptured(picture: ByteArray) {





        Toast.makeText(this, "RUN SAVE.", Toast.LENGTH_SHORT).show()
        val bitmap = BitmapFactory.decodeByteArray(picture, 0, picture.size)
        val squareBitmap = ThumbnailUtils.extractThumbnail(bitmap, getScreenWidth(), getScreenWidth())


        ivPreview.setImageBitmap(squareBitmap)

        saveImage(squareBitmap)


        /// save image


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
                tvClassification.text = output
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
                tvClassification.text = output
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
                tvClassification.text = output
            }



        } else if (options.selectedItem == "MnasNet") {

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
                tvClassification.text = output
            }

        }else {
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
                tvClassification.text = output
            }

        }



        //// save image



        //    saveImage(squareBitmap)


    }


    private fun saveImage(finalBitmap: Bitmap) {

        val root = Environment.getExternalStorageDirectory().toString()
        val myDir = File(root + "/capture_photo")
        myDir.mkdirs()
        val generator = Random()
        var n = 10000
        n = generator.nextInt(n)
        val OutletFname = "Image-$n.jpg"
       // Toast.makeText(this, "RUN Save image root " + root, Toast.LENGTH_SHORT).show()
        val file = File(myDir, OutletFname)
        if (file.exists()) file.delete()
        try {
            val out = FileOutputStream(file)
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            imagePath = file.absolutePath
           // Toast.makeText(this, "RUN Save gambar " + imagePath, Toast.LENGTH_SHORT).show()

            out.flush()
            out.close()


        } catch (e: Exception) {
            e.printStackTrace()

        }

    }

    private fun getScreenWidth(): Int {
        val displayMetrics = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(displayMetrics)
        return displayMetrics.widthPixels
    }
}
