package com.ffztest.opencvtest

import android.R.attr.src
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.ffztest.opencvtest.databinding.ActivityMainBinding
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc


class MainActivity : AppCompatActivity() {

    companion object {
        const val TAG = "openCV"
    }

    private lateinit var binding: ActivityMainBinding

    private var srcBitmap: Bitmap? = null

    private var resultBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        initClickListener()
    }

    private fun initClickListener() {
        binding.apply {
            btnLoadImage.setOnClickListener {
                srcBitmap =
                    BitmapFactory.decodeResource(this@MainActivity.resources, R.drawable.astronaut)
                srcBitmap?.let { image ->
                    ivSrc.setImageBitmap(image)
                }
            }

            btnHeTest.setOnClickListener {
                resultBitmap = imageHE()

                resultBitmap?.let { result ->
                    ivResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            btnGrayTest.setOnClickListener {
                resultBitmap = toGray()

                resultBitmap?.let { result ->
                    ivResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            btnClaheTest.setOnClickListener {
                resultBitmap = imageCLAHE()

                resultBitmap?.let { result ->
                    ivClaheResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            btnLaprasTest.setOnClickListener {
                resultBitmap = imageLapras()

                resultBitmap?.let { result ->
                    ivResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            btnLogTest.setOnClickListener {
                resultBitmap = imageLog()

                resultBitmap?.let { result ->
                    ivResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            btnGammaTest.setOnClickListener {
                resultBitmap = imageGamma()

                resultBitmap?.let { result ->
                    ivResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
            }

            slider.addOnChangeListener { _, value, _ ->
                resultBitmap = imageCLAHE(value.toDouble())

                resultBitmap?.let { result ->
                    ivClaheResult.apply {
                        setImageResource(0)
                        setImageBitmap(result)
                    }
                }
                tvSliderNum.text = value.toString()
            }
        }
    }

    /**
     * 图像灰度化
     */
    private fun toGray(): Bitmap? {
        val src = Mat()
        val dst = Mat()

        val bitmap = srcBitmap?.copy(Bitmap.Config.ARGB_8888, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            Utils.bitmapToMat(it, src)
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY)
            Utils.matToBitmap(dst, it)
        }

        src.release()
        dst.release()
        return bitmap
    }

    /**
     * 均衡化算法
     */
    private fun imageHE(): Bitmap? {
        val src = Mat()
        val splitArray = ArrayList<Mat>()
        val dst = Mat()

        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            Utils.bitmapToMat(it, src)
            Core.split(src, splitArray)
            Log.d(TAG, "splitArray.size = ${splitArray.size}")
            for (item in splitArray) {
                Imgproc.equalizeHist(item, item)
            }
            Core.merge(splitArray, dst)
            Utils.matToBitmap(dst, it)
        }

        src.release()
        dst.release()
        return bitmap
    }

    /**
     * 自适应均衡化算法---使用默认裁剪值
     */
    private fun imageCLAHE(): Bitmap? {
        val src = Mat()
        val splitArray = ArrayList<Mat>()
        val dst = Mat()
        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            val clahe = Imgproc.createCLAHE()
            Utils.bitmapToMat(it, src)
            Core.split(src, splitArray)
            Log.d(TAG, "splitArray.size = ${splitArray.size}")
            for (item in splitArray) {
                clahe.apply(item, item)
            }
            Core.merge(splitArray, dst)
            Utils.matToBitmap(dst, it)
        }

        src.release()
        dst.release()

        return bitmap
    }

    /**
     * 自适应均衡化算法---变化的
     */
    private fun imageCLAHE(value: Double): Bitmap? {
        val src = Mat()
        val splitArray = ArrayList<Mat>()
        val dst = Mat()
        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            val clahe = Imgproc.createCLAHE(value)
            Utils.bitmapToMat(it, src)
            Core.split(src, splitArray)
            Log.d(TAG, "splitArray.size = ${splitArray.size}")
            for (item in splitArray) {
                clahe.apply(item, item)
            }
            Core.merge(splitArray, dst)
            Utils.matToBitmap(dst, it)
        }

        src.release()
        dst.release()

        return bitmap
    }

    /**
     * 拉普拉斯算法增强
     */
    private fun imageLapras(): Bitmap? {
        val src = Mat()
        val dst = Mat()
        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            Utils.bitmapToMat(it, src)
            val kernel = floatArrayOf(0f, 0f, 0f, -1f, 5f, -1f, 0f, 0f, 0f)
            val kernelMat = Mat(3, 3, CvType.CV_32FC1)
            kernelMat.put(0, 0, kernel)
            Imgproc.filter2D(src, dst, CvType.CV_8UC3, kernelMat)
            Utils.matToBitmap(dst, it)
        }

        return bitmap
    }

    /**
     * 对数变换增强
     */
    private fun imageLog(): Bitmap? {
        val src = Mat()
        var dst = Mat()
        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            Utils.bitmapToMat(it, src)
            dst = Mat(src.size(), CvType.CV_32FC3)
            Core.add(src, Scalar(5.0, 5.0, 5.0), src)
            src.convertTo(src, CvType.CV_32F)
            Core.log(src, dst)
            //        Core.multiply(imageLog, new Scalar(3,3,3), imageLog);
            Core.normalize(dst, dst, 0.0, 255.0, Core.NORM_MINMAX)
            Core.convertScaleAbs(dst, dst)
            Utils.matToBitmap(dst, it)
        }

        return bitmap
    }

    /**
     * 伽马变化增强
     */
    private fun imageGamma(): Bitmap? {
        val src = Mat()
        var dst = Mat()
        val bitmap = srcBitmap?.copy(Bitmap.Config.RGB_565, true)

        if (null == bitmap) {
            showToast()
            return null
        }

        bitmap.let {
            Utils.bitmapToMat(it, src)
            dst = src.clone()
            dst.convertTo(dst, CvType.CV_32F)

            Core.pow(dst, 4.0, dst)

            Core.normalize(dst, dst, 0.0, 255.0, Core.NORM_MINMAX)
            Core.convertScaleAbs(dst, dst)
            Utils.matToBitmap(dst, it)
        }
        return bitmap
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            // TODO Auto-generated method stub
            when (status) {
                SUCCESS -> Log.i(TAG, "成功加载")
                else -> {
                    super.onManagerConnected(status)
                    Log.i(TAG, "加载失败")
                }
            }
        }
    }


    private fun showToast(msg: String = "please load picture first") {
        Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
    }
}