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


public class ztest extends AppCompatActivity implements AdapterView.OnItemSelectedListener {


    PyObject pyattr;
    TextView X1,y1,p1,the,ts,nsp,nfp,sdsp,sdfp;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin,vicky_spin1,vicky_spin2;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ztest);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("Z - t e s t");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);
        Spinner vicky_spin1 = (Spinner) findViewById(R.id.spinner6);

        ArrayAdapter<String> my_adpt1 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.ts));
        my_adpt1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin1.setAdapter(my_adpt1);

        vicky_spin1.setOnItemSelectedListener(this);

        Spinner vicky_spin2 = (Spinner) findViewById(R.id.spinner7);

        ArrayAdapter<String> my_adpt2 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin2.setAdapter(my_adpt2);

        Spinner vicky_spin3 = (Spinner) findViewById(R.id.spinner27);

        ArrayAdapter<String> my_adpt3 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin3.setAdapter(my_adpt3);

        Spinner vicky_spin4 = (Spinner) findViewById(R.id.spinner37);

        ArrayAdapter<String> my_adpt4 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
        my_adpt4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin4.setAdapter(my_adpt4);

        Spinner vicky_spin5 = (Spinner) findViewById(R.id.spinner471);

        ArrayAdapter<String> my_adpt5 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.yn));
        my_adpt5.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin5.setAdapter(my_adpt5);

        Spinner vicky_spin6 = (Spinner) findViewById(R.id.spinner47);

        ArrayAdapter<String> my_adpt6 = new ArrayAdapter<String>(ztest.this, R.layout.vicky_spinner, getResources().getStringArray(R.array.tlr));
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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.ztest;
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
            }, 40000);
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String pos=String.valueOf(position);
        LinearLayout ly1=(LinearLayout) findViewById(R.id.ly1);
        LinearLayout ly2=(LinearLayout) findViewById(R.id.ly2);
        LinearLayout ly3=(LinearLayout) findViewById(R.id.ly3);
        LinearLayout ly4=(LinearLayout) findViewById(R.id.ly4);
        LinearLayout ly5=(LinearLayout) findViewById(R.id.ly5);
        LinearLayout ly6=(LinearLayout) findViewById(R.id.ly6);

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

        LinearLayout ly31=(LinearLayout) findViewById(R.id.ly31);
        LinearLayout ly32=(LinearLayout) findViewById(R.id.ly32);
        LinearLayout ly33=(LinearLayout) findViewById(R.id.ly33);
        LinearLayout ly34=(LinearLayout) findViewById(R.id.ly34);
        LinearLayout ly35=(LinearLayout) findViewById(R.id.ly35);

        LinearLayout ly41=(LinearLayout) findViewById(R.id.ly41);
        LinearLayout ly42=(LinearLayout) findViewById(R.id.ly42);
        LinearLayout ly43=(LinearLayout) findViewById(R.id.ly43);
        LinearLayout ly44=(LinearLayout) findViewById(R.id.ly44);
        LinearLayout ly45=(LinearLayout) findViewById(R.id.ly45);
        LinearLayout ly46=(LinearLayout) findViewById(R.id.ly46);
        LinearLayout ly47=(LinearLayout) findViewById(R.id.ly47);
        LinearLayout ly48=(LinearLayout) findViewById(R.id.ly48);
        LinearLayout ly49=(LinearLayout) findViewById(R.id.ly49);
        if (pos.equals("0")){
            ly1.setVisibility(View.VISIBLE);
            ly2.setVisibility(View.VISIBLE);
            ly3.setVisibility(View.VISIBLE);
            ly4.setVisibility(View.VISIBLE);
            ly5.setVisibility(View.VISIBLE);
            ly6.setVisibility(View.VISIBLE);
        }
        else{
            ly1.setVisibility(View.GONE);
            ly2.setVisibility(View.GONE);
            ly3.setVisibility(View.GONE);
            ly4.setVisibility(View.GONE);
            ly5.setVisibility(View.GONE);
            ly6.setVisibility(View.GONE);
        }

        if (pos.equals("2")){
            ly31.setVisibility(View.VISIBLE);
            ly32.setVisibility(View.VISIBLE);
            ly33.setVisibility(View.VISIBLE);
            ly34.setVisibility(View.VISIBLE);
            ly35.setVisibility(View.VISIBLE);

        }
        else{
            ly31.setVisibility(View.GONE);
            ly32.setVisibility(View.GONE);
            ly33.setVisibility(View.GONE);
            ly34.setVisibility(View.GONE);
            ly35.setVisibility(View.GONE);

        }

        if (pos.equals("1")){
            ly21.setVisibility(View.VISIBLE);
            ly22.setVisibility(View.VISIBLE);
            ly23.setVisibility(View.VISIBLE);
            ly24.setVisibility(View.VISIBLE);
            ly25.setVisibility(View.VISIBLE);
            ly26.setVisibility(View.VISIBLE);
            ly27.setVisibility(View.VISIBLE);
            ly28.setVisibility(View.VISIBLE);
            ly29.setVisibility(View.VISIBLE);
            ly210.setVisibility(View.VISIBLE);
        }
        else{
            ly21.setVisibility(View.GONE);
            ly22.setVisibility(View.GONE);
            ly23.setVisibility(View.GONE);
            ly24.setVisibility(View.GONE);
            ly25.setVisibility(View.GONE);
            ly26.setVisibility(View.GONE);
            ly27.setVisibility(View.GONE);
            ly28.setVisibility(View.GONE);
            ly29.setVisibility(View.GONE);
            ly210.setVisibility(View.GONE);
        }
        if (pos.equals("3")){
            ly41.setVisibility(View.VISIBLE);
            ly42.setVisibility(View.VISIBLE);
            ly43.setVisibility(View.VISIBLE);
            ly44.setVisibility(View.VISIBLE);
            ly45.setVisibility(View.VISIBLE);
            ly46.setVisibility(View.VISIBLE);
            ly47.setVisibility(View.VISIBLE);
            ly48.setVisibility(View.VISIBLE);
            ly49.setVisibility(View.VISIBLE);
        }
        else{
            ly41.setVisibility(View.GONE);
            ly42.setVisibility(View.GONE);
            ly43.setVisibility(View.GONE);
            ly44.setVisibility(View.GONE);
            ly45.setVisibility(View.GONE);
            ly46.setVisibility(View.GONE);
            ly47.setVisibility(View.GONE);
            ly48.setVisibility(View.GONE);
            ly49.setVisibility(View.GONE);
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }


    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(ztest.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        vicky_spin1 = (Spinner) findViewById(R.id.spinner6);
        int str5 = vicky_spin1.getSelectedItemPosition();
        int k=0;
        if (str5==0) {
            X1 = (TextView) findViewById(R.id.ns);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k = 1;
                y1 = (TextView) findViewById(R.id.los);
                p1 = (TextView) findViewById(R.id.sd);
                the = (TextView) findViewById(R.id.mos);
                ts = (TextView) findViewById(R.id.pm);
                vicky_spin = (Spinner) findViewById(R.id.spinner7);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                String str6 = ts.getText().toString();
                String str7 = p1.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("ztest", str5, str4, str2, str1, str7, str3, str6);
            }
        }

        else if (str5==1) {
            X1 = (TextView) findViewById(R.id.mfps);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k = 1;
                y1 = (TextView) findViewById(R.id.los2);
                p1 = (TextView) findViewById(R.id.msps);
                the = (TextView) findViewById(R.id.mfp);
                ts = (TextView) findViewById(R.id.msp);
                nfp = (TextView) findViewById(R.id.nfps);
                nsp = (TextView) findViewById(R.id.nsps);
                sdfp = (TextView) findViewById(R.id.sdfp);
                sdsp = (TextView) findViewById(R.id.sdsp);
                vicky_spin = (Spinner) findViewById(R.id.spinner27);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                String str6 = ts.getText().toString();
                String str7 = nfp.getText().toString();
                String str8 = nsp.getText().toString();
                String str9 = p1.getText().toString();
                String str10 = sdfp.getText().toString();
                String str11 = sdsp.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("ztest", str5, str4, str2, str1, str9, str3, str6, str7, str8, str10, str11);
            }
        }

        else if (str5==2) {
            X1 = (TextView) findViewById(R.id.nps);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k = 1;
                y1 = (TextView) findViewById(R.id.los3);
                p1 = (TextView) findViewById(R.id.sp);
                the = (TextView) findViewById(R.id.pp);
                vicky_spin = (Spinner) findViewById(R.id.spinner37);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                String str6 = p1.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("ztest", str5, str4, str2, str1, str6, str3);
            }
        }

        else if (str5==3) {
            X1 = (TextView) findViewById(R.id.nfpsp);
            String str1 = X1.getText().toString();
            if (!str1.equals("")) {
                k = 1;
                y1 = (TextView) findViewById(R.id.los4);
                p1 = (TextView) findViewById(R.id.nspsp);
                the = (TextView) findViewById(R.id.spsp);
                ts = (TextView) findViewById(R.id.spfp);
                nfp = (TextView) findViewById(R.id.pfp);
                nsp = (TextView) findViewById(R.id.psp);
                vicky_spin = (Spinner) findViewById(R.id.spinner47);
                vicky_spin2 = (Spinner) findViewById(R.id.spinner471);
                String str2 = y1.getText().toString();
                String str3 = the.getText().toString();
                int str4 = vicky_spin.getSelectedItemPosition();
                String str6 = ts.getText().toString();
                String str7 = nfp.getText().toString();
                String str8 = nsp.getText().toString();
                int str9 = vicky_spin2.getSelectedItemPosition();
                String str10 = p1.getText().toString();
                pyattr = splash_screen.pyobj.callAttr("ztest", str5, str4, str2, str1, str10, str3, str6, str7, str8, str9);
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