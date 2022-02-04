package com.ML_Calculator.domlwithv_vignesh;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.android.gms.ads.MobileAds;

public class splash_screen extends AppCompatActivity {

    ImageView gif;
    public static Python py;
    public static PyObject pyobj;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_splash_screen);
        MobileAds.initialize(this);
        LinearLayout re=(LinearLayout) findViewById(R.id.linearLayout4);

        re.post(new Runnable(){
            public void run(){
                int wd = re.getWidth()-791;
                String str=String.valueOf(wd);
                TextView tx=(TextView)findViewById(R.id.Tittle);
                tx.setTextSize(TypedValue.COMPLEX_UNIT_SP, wd);
            }
        });

        gif=findViewById(R.id.gif);

        Glide.with(this).asGif().load(R.raw.finalgif).into(gif);


        new Handler().postDelayed(new Runnable() {
            @Override public void run() {
        if (!Python.isStarted()) {
                     Python.start(new AndroidPlatform(splash_screen.this));
                    py=Python.getInstance();
                    pyobj=py.getModule("linear");
            Intent intent= new Intent(splash_screen.this,MainActivity.class);
            startActivity(intent);
            finish();
        }
            }
        },5000);



    }
}