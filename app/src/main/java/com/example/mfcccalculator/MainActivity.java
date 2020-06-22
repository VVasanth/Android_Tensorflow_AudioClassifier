package com.example.mfcccalculator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        int mNumFrames;
        int[] mFrameGains;
        int mFileSize;
        int mSampleRate;
        int mChannels;

        File externalStorage = Environment.getExternalStorageDirectory();

        File sourceFile = new File(externalStorage.getAbsolutePath() , "/1995-1826-0003.wav");
        WavFile wavFile = null;
        try {
            wavFile = WavFile.openWavFile(sourceFile);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (WavFileException e) {
            e.printStackTrace();
        }

    }

}