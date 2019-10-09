package com.mine.mytensorflow


import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.support.v4.app.Fragment
import android.support.v7.app.AlertDialog
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import kotlinx.android.synthetic.main.fragment_classification.*
import java.io.IOException
import android.R.attr.bitmap
import android.media.ExifInterface



class ClassificationFragment : Fragment() {

    private val SELECT_PICTURE = 100
    private val CAMERA_REQUEST = 101

    private var filepath: Uri? = null

    private var original_image: Bitmap? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_classification, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        takePicture.setOnClickListener { startDialog() }
        initSpinner()
    }

    private fun initSpinner() {
        options.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
                Toast.makeText(context, "" + options.selectedItem, Toast.LENGTH_LONG).show()
            }
            override fun onNothingSelected(parent: AdapterView<*>) {
            }
        }
    }

    private fun startDialog() {
        context?.let {
            val myAlertDialog = AlertDialog.Builder(
                it
            )
            myAlertDialog.setTitle("Take Pictures Option")
            myAlertDialog.setMessage("How do you want to set your picture?")

            myAlertDialog.setPositiveButton(
                "Gallery"
            ) { arg0, arg1 ->
                showImageChooser()
            }

            myAlertDialog.setNegativeButton(
                "Camera"
            ) { arg0, arg1 ->
                dispatchTakePictureIntent()
            }
            myAlertDialog.show()
        }

    }

    fun showImageChooser() {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == SELECT_PICTURE
            && resultCode == Activity.RESULT_OK
            && data != null
            && data.data != null) {

            displaySelectedImage(data)
        }
        if (requestCode == CAMERA_REQUEST && resultCode == AppCompatActivity.RESULT_OK && data!=null) {
            val imageBitmap = data.extras?.get("data") as Bitmap

            imgResult.setImageBitmap(imageBitmap.rotate(90))

        }

    }


    private fun displaySelectedImage(data: Intent) {
        filepath = data.data
        try {
            original_image = MediaStore.Images.Media.getBitmap(activity?.contentResolver, filepath)
            imgResult.setImageBitmap(original_image)


        } catch (e: IOException) {
            Log.d("MainActivity", "Error : $e")
        }

    }

    private fun dispatchTakePictureIntent() {
        context?.let {
            Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
                takePictureIntent.resolveActivity(it.packageManager)?.also {
                   startActivityForResult(takePictureIntent, CAMERA_REQUEST)
                }
            }
        }

    }



}
