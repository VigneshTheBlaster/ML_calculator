package com.ML_Calculator.domlwithv_vignesh;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.text.method.ScrollingMovementMethod;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.ArrayAdapter;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import com.bumptech.glide.Glide;
import com.chaquo.python.PyObject;
import com.google.android.gms.ads.AdRequest;
import com.google.android.gms.ads.AdView;

import java.util.List;

public class ftest extends AppCompatActivity {

    PyObject pyattr;
    TextView X1,y1,p1,the,ts;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin,vicky_spin1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ftest);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("F - t e s t");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);

        Spinner vicky_spin5 = (Spinner) findViewById(R.id.spinner471);

        ArrayAdapter<String> my_adpt5 = new ArrayAdapter<String>(ftest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.yn));
        my_adpt5.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin5.setAdapter(my_adpt5);

        Spinner vicky_spin6 = (Spinner) findViewById(R.id.spinner47);

        ArrayAdapter<String> my_adpt6 = new ArrayAdapter<String>(ftest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt6.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin6.setAdapter(my_adpt6);
    }

    public void ply(View v){
        FrameLayout re1=(FrameLayout)findViewById(R.id.re1);
        vv = (VideoView) findViewById(R.id.videoView);
        if (pls==1){
            vv.stopPlayback();
            re1.setVisibility(View.GONE);
            demo.setBackgroundResource(R.drawable.play_icon);
            pls=0;
        }
        else {
            pls = 1;
            re1.setVisibility(View.VISIBLE);
            String str = "android.resource://" + getPackageName() + "/" + R.raw.ftest;
            Uri uri = Uri.parse(str);
            vv.setVideoURI(uri);
            vv.start();
            demo.setBackgroundResource(R.drawable.pauses);
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    re1.setVisibility(View.GONE);
                    demo.setBackgroundResource(R.drawable.play_icon);
                    pls = 0;
                }
            }, 39000);
        }
    }


    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(ftest.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        X1 = (TextView) findViewById(R.id.nfpsp);
        String str1 = X1.getText().toString();
        if (!str1.equals("")) {
            gif = findViewById(R.id.gif);
            LinearLayout sc = (LinearLayout) findViewById(R.id.scrollLayout);
            sc.setVisibility(View.GONE);
            gif.setVisibility(View.VISIBLE);
            gif();
            y1 = (TextView) findViewById(R.id.los4);
            p1 = (TextView) findViewById(R.id.nspsp);
            the = (TextView) findViewById(R.id.mfps);
            ts = (TextView) findViewById(R.id.msps);
            vicky_spin = (Spinner) findViewById(R.id.spinner47);
            vicky_spin1 = (Spinner) findViewById(R.id.spinner471);
            String str2 = y1.getText().toString();
            String str3 = the.getText().toString();
            int str4 = vicky_spin.getSelectedItemPosition();
            int str5 = vicky_spin1.getSelectedItemPosition();
            String str6 = ts.getText().toString();
            String str7 = p1.getText().toString();
            pyattr = splash_screen.pyobj.callAttr("ftest", str4, str5, str2, str1, str7, str3, str6);


            TextView out1 = (TextView) findViewById(R.id.pre1);
            TextView textViewDD = (TextView) findViewById(R.id.pre1);
            textViewDD.setMovementMethod(new ScrollingMovementMethod());
            textViewDD.setOnTouchListener(new View.OnTouchListener() {
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                    v.getParent().requestDisallowInterceptTouchEvent(true);
                    return false;
                }
            });
            List<PyObject> val = pyattr.asList();
            out1.setText(val.get(0).toString());
            String err = val.get(1).toString();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    if (err.equals("error")) {
                        sc.getLayoutParams().height = 280;
                        out1.getLayoutParams().height = 190;
                    } else {
                        sc.getLayoutParams().height = 720;
                        out1.getLayoutParams().height = 670;
                    }
                    sc.setVisibility(View.VISIBLE);
                    gif.setVisibility(View.GONE);
                }
            }, 3000);

        }
        else{
            Toast.makeText(this,"Please Enter appropriate data darling",Toast.LENGTH_LONG).show();
        }

    }

}