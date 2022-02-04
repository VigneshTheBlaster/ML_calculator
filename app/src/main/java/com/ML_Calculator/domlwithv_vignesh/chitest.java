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
import android.widget.AdapterView;
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

public class chitest extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    PyObject pyattr;
    TextView X1,y1,p1,the,ts;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;

    Spinner vicky_spin,vicky_spin1,vicky_spin2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chitest);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("chi-square t e s t");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);

        Spinner vicky_spin1 = (Spinner) findViewById(R.id.spinner6);

        ArrayAdapter<String> my_adpt1 = new ArrayAdapter<String>(chitest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tof));
        my_adpt1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin1.setAdapter(my_adpt1);
        vicky_spin1.setOnItemSelectedListener(this);

        Spinner vicky_spin2 = (Spinner) findViewById(R.id.spinner27);
        ArrayAdapter<String> my_adpt2 = new ArrayAdapter<String>(chitest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.db));
        my_adpt2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin2.setAdapter(my_adpt2);
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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.chi;
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
            }, 27000);
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

        String pos=String.valueOf(position);
        LinearLayout ly1=(LinearLayout) findViewById(R.id.ly1);
        LinearLayout ly2=(LinearLayout) findViewById(R.id.ly2);
        LinearLayout ly21=(LinearLayout) findViewById(R.id.ly21);
        LinearLayout ly22=(LinearLayout) findViewById(R.id.ly22);
        LinearLayout ly23=(LinearLayout) findViewById(R.id.ly23);
        LinearLayout ly24=(LinearLayout) findViewById(R.id.ly24);

        if (pos.equals("0")) {
            ly1.setVisibility(View.VISIBLE);
            ly2.setVisibility(View.VISIBLE);
        }
        else{
            ly1.setVisibility(View.GONE);
            ly2.setVisibility(View.GONE);
        }

        if (pos.equals("1")) {
            ly21.setVisibility(View.VISIBLE);
            ly22.setVisibility(View.VISIBLE);
            ly23.setVisibility(View.VISIBLE);
            ly24.setVisibility(View.VISIBLE);
        }
        else{
            ly21.setVisibility(View.GONE);
            ly22.setVisibility(View.GONE);
            ly23.setVisibility(View.GONE);
            ly24.setVisibility(View.GONE);
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(chitest.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        vicky_spin2 = (Spinner) findViewById(R.id.spinner6);
        int str8 = vicky_spin2.getSelectedItemPosition();
        int k=0;
        if (str8==0) {
            X1 = (TextView) findViewById(R.id.X1);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k=1;
                y1 = (TextView) findViewById(R.id.los);
                String str2 = y1.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("chitest", str8, str2, str1);
            }
        }
        else if (str8==1) {
            p1 = (TextView) findViewById(R.id.obs);
            String str7 = p1.getText().toString();
            if (!str7.equals("")) {
                k=1;
                y1 = (TextView) findViewById(R.id.los1);
                the = (TextView) findViewById(R.id.exp);
                vicky_spin = (Spinner) findViewById(R.id.spinner27);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                pyattr = splash_screen.pyobj.callAttr("chitest", str8, str2, str7, str4, str3);
            }

        }

        if (k==1) {

            gif=findViewById(R.id.gif);
            LinearLayout sc=(LinearLayout)findViewById(R.id.scrollLayout);
            sc.setVisibility(View.GONE);
            gif.setVisibility(View.VISIBLE);
            gif();

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