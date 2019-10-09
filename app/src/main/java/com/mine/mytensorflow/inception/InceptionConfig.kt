package com.mine.mytensorflow.inception

import android.app.Activity
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.ArrayList
import java.util.Arrays
import java.util.Collections

/**
 * The most of those information can be found in GTSRB_TensorFlow_MobileNet.ipynb
 */
object InceptionConfig {
    var MODEL_FILENAME = "inceptionv3_graph.lite"

    val INPUT_IMG_SIZE_WIDTH = 299
    val INPUT_IMG_SIZE_HEIGHT = 299
    val FLOAT_TYPE_SIZE = 4
    val PIXEL_SIZE = 3
    val MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE
    val IMAGE_MEAN = 0
    val IMAGE_STD = 255.0f

    //This list can be taken from notebooks/output/labels_readable.txt
    val OUTPUT_LABELS = Collections.unmodifiableList(
        Arrays.asList(
            "appleblackrot",
            "applecedarapplerust",
            "applehealthy",
            "applescab",
            "blueberryhealthy",
            "cherry including sour powderymildew",
            "cherry including sour healthy",
            "corn maize cercosporaleafspotgrayleafspot",
            "corn maize commonrust",
            "corn maize northernleafblight",
            "corn maize healthy",
            "grapeblackrot",
            "grapeesca blackmeasles",
            "grapeleafblight isariopsisleafspot",
            "grapehealthy",
            "orangehaunglongbing citrusgreening",
            "peachbacterialspot",
            "peachhealthy",
            "pepperbellbacterialspot",
            "pepperbellhealthy",
            "potatoearlyblight",
            "potatolateblight",
            "potatohealthy",
            "raspberryhealthy",
            "soybeanhealthy",
            "squashpowderymildew",
            "strawberryleafscorch",
            "strawberryhealthy",
            "tomatobacterialspot",
            "tomatoearlyblight",
            "tomatolateblight",
            "tomatoleafmold",
            "tomatoseptorialeafspot",
            "tomatospidermitestwospottedspidermite",
            "tomatotargetspot",
            "tomatoyellowleafcurlvirus",
            "tomatohealthy",
            "tomatomosaicvirus",
            "iccv09data"
        )
    )


    @Throws(IOException::class)
    private fun loadLabelList(activity: Activity, path:String): List<String> {
        val labels = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(activity.assets.open(path)))

        while ( reader.readLine() != null) {
            labels.add(reader.readLine())
        }
        reader.close()
        return labels
    }

    val MAX_CLASSIFICATION_RESULTS = 1
    val CLASSIFICATION_THRESHOLD = 0.1f
}