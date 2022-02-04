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

public class ttest extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    PyObject pyattr;
    TextView X1,y1,p1,the,ts,nsp,nfp,sdsp,sdfp;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin,vicky_spin1,vicky_spin2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ttest);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("t - t e s t");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);

        Spinner vicky_spin1 = (Spinner) findViewById(R.id.spinner6);

        ArrayAdapter<String> my_adpt1 = new ArrayAdapter<String>(ttest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tt));
        my_adpt1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin1.setAdapter(my_adpt1);
        vicky_spin1.setOnItemSelectedListener(this);

        Spinner vicky_spin2 = (Spinner) findViewById(R.id.spinner7);
        ArrayAdapter<String> my_adpt2 = new ArrayAdapter<String>(ttest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin2.setAdapter(my_adpt2);

        Spinner vicky_spin5 = (Spinner) findViewById(R.id.spinner17);
        ArrayAdapter<String> my_adpt5 = new ArrayAdapter<String>(ttest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.yn));
        my_adpt5.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin5.setAdapter(my_adpt5);

        Spinner vicky_spin6 = (Spinner) findViewById(R.id.spinner471);
        ArrayAdapter<String> my_adpt6 = new ArrayAdapter<String>(ttest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.ds));
        my_adpt6.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin6.setAdapter(my_adpt6);
        vicky_spin6.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String pos=String.valueOf(position);
                LinearLayout ly27=(LinearLayout) findViewById(R.id.ly27);
                LinearLayout ly28=(LinearLayout) findViewById(R.id.ly28);
                LinearLayout ly281=(LinearLayout) findViewById(R.id.ly281);
                if (pos.equals("1") || pos.equals("2")){
                    ly27.setVisibility(View.VISIBLE);
                    ly28.setVisibility(View.VISIBLE);
                }
                else{
                    ly27.setVisibility(View.GONE);
                    ly28.setVisibility(View.GONE);
                }

                if (pos.equals("3")){
                    ly281.setVisibility(View.VISIBLE);
                }
                else{
                    ly281.setVisibility(View.GONE);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });



        Spinner vicky_spin3 = (Spinner) findViewById(R.id.spinner27);
        ArrayAdapter<String> my_adpt3 = new ArrayAdapter<String>(ttest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin3.setAdapter(my_adpt3);


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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.ttest;
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

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String p=parent.getSelectedItem().toString();
        String pos=String.valueOf(position);
        LinearLayout ly1=(LinearLayout) findViewById(R.id.ly1);
        LinearLayout ly2=(LinearLayout) findViewById(R.id.ly2);
        LinearLayout ly3=(LinearLayout) findViewById(R.id.ly3);
        LinearLayout ly4=(LinearLayout) findViewById(R.id.ly4);
        LinearLayout ly5=(LinearLayout) findViewById(R.id.ly5);
        LinearLayout ly6=(LinearLayout) findViewById(R.id.ly6);
        LinearLayout ly7=(LinearLayout) findViewById(R.id.ly7);

        LinearLayout ly21=(LinearLayout) findViewById(R.id.ly21);
        LinearLayout ly22=(LinearLayout) findViewById(R.id.ly22);
        LinearLayout ly23=(LinearLayout) findViewById(R.id.ly23);
        LinearLayout ly24=(LinearLayout) findViewById(R.id.ly24);
        LinearLayout ly25=(LinearLayout) findViewById(R.id.ly25);
        LinearLayout ly26=(LinearLayout) findViewById(R.id.ly26);
        LinearLayout ly27=(LinearLayout) findViewById(R.id.ly27);
        LinearLayout ly28=(LinearLayout) findViewById(R.id.ly28);
        LinearLayout ly29=(LinearLayout) findViewById(R.id.ly29);
        LinearLayout ly210=(LinearLayout) findViewById(R.id.ly210);
        LinearLayout ly211=(LinearLayout) findViewById(R.id.ly211);
        LinearLayout ly281=(LinearLayout) findViewById(R.id.ly281);



        if (p.equals("Single mean")){
            ly1.setVisibility(View.VISIBLE);
            ly2.setVisibility(View.VISIBLE);
            ly3.setVisibility(View.VISIBLE);
            ly4.setVisibility(View.VISIBLE);
            ly5.setVisibility(View.VISIBLE);
            ly6.setVisibility(View.VISIBLE);
            ly7.setVisibility(View.VISIBLE);
            ly27.setVisibility(View.GONE);
            ly28.setVisibility(View.GONE);
            ly281.setVisibility(View.GONE);
        }
        else{
            ly1.setVisibility(View.GONE);
            ly2.setVisibility(View.GONE);
            ly3.setVisibility(View.GONE);
            ly4.setVisibility(View.GONE);
            ly5.setVisibility(View.GONE);
            ly6.setVisibility(View.GONE);
            ly7.setVisibility(View.GONE);

        }

        if (pos.equals("1")){
            ly21.setVisibility(View.VISIBLE);
            ly22.setVisibility(View.VISIBLE);
            ly23.setVisibility(View.VISIBLE);
            ly24.setVisibility(View.VISIBLE);
            ly25.setVisibility(View.VISIBLE);
            ly26.setVisibility(View.VISIBLE);
            ly29.setVisibility(View.VISIBLE);
            ly210.setVisibility(View.VISIBLE);
            ly211.setVisibility(View.VISIBLE);
        }
        else{
            ly21.setVisibility(View.GONE);
            ly22.setVisibility(View.GONE);
            ly23.setVisibility(View.GONE);
            ly24.setVisibility(View.GONE);
            ly25.setVisibility(View.GONE);
            ly26.setVisibility(View.GONE);
            ly29.setVisibility(View.GONE);
            ly210.setVisibility(View.GONE);
            ly211.setVisibility(View.GONE);
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(ttest.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        vicky_spin1 = (Spinner) findViewById(R.id.spinner6);
        int str5 = vicky_spin1.getSelectedItemPosition();
        int k=0;
        if (str5==0) {
            X1 = (TextView) findViewById(R.id.ns);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k=1;
                y1 = (TextView) findViewById(R.id.los);
                p1 = (TextView) findViewById(R.id.sd);
                the = (TextView) findViewById(R.id.mos);
                ts = (TextView) findViewById(R.id.pm);
                vicky_spin = (Spinner) findViewById(R.id.spinner7);
                vicky_spin2 = (Spinner) findViewById(R.id.spinner17);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                int str8 = vicky_spin2.getSelectedItemPosition();
                String str6 = ts.getText().toString();
                String str7 = p1.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("ttest", str5, str4, str2, str1, str3, str6, str7, str8);
            }
        }

        else if (str5==1) {
            X1 = (TextView) findViewById(R.id.mfps);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k=1;
                y1 = (TextView) findViewById(R.id.los2);
                p1 = (TextView) findViewById(R.id.msps);
                the = (TextView) findViewById(R.id.mfp);
                ts = (TextView) findViewById(R.id.msp);
                nfp = (TextView) findViewById(R.id.nfps);
                nsp = (TextView) findViewById(R.id.nsps);
                vicky_spin = (Spinner) findViewById(R.id.spinner27);
                vicky_spin2 = (Spinner) findViewById(R.id.spinner471);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                int str44 = vicky_spin2.getSelectedItemPosition();
                String str6 = ts.getText().toString();
                String str7 = nfp.getText().toString();
                String str8 = nsp.getText().toString();
                String str9 = p1.getText().toString();

                if (str44 == 1 || str44 == 2 || str44 == 0) {
                    sdfp = (TextView) findViewById(R.id.sdfp);
                    sdsp = (TextView) findViewById(R.id.sdsp);
                    String str10 = sdfp.getText().toString();
                    String str11 = sdsp.getText().toString();
                    pyattr = splash_screen.pyobj.callAttr("ttest", str5, str4, str2, str44, str1, str9, str7, str8, str3, str6, str10, str11);
                } else {
                    sdfp = (TextView) findViewById(R.id.sdp);
                    String str10 = sdfp.getText().toString();
                    pyattr = splash_screen.pyobj.callAttr("ttest", str5, str4, str2, str44, str1, str9, str7, str8, str3, str6, str10);
                }
            }
        }
        if(k==1) {

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