package com.ML_Calculator.domlwithv_vignesh;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.text.method.ScrollingMovementMethod;
import android.util.Base64;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
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

public class hierc extends AppCompatActivity {

    PyObject pyattr;
    TextView X1,p1,ts;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin,vicky_spin1,vicky_spin2,vicky_spin3,vicky_spin4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_hierc);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("hierarchical cluster");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);
        Spinner vicky_spin4 = (Spinner) findViewById(R.id.spinner6);

        ArrayAdapter<String> my_adpt4 = new ArrayAdapter<String>(hierc.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.dm));
        my_adpt4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin4.setAdapter(my_adpt4);

        Spinner vicky_spin1 = (Spinner) findViewById(R.id.spinner2);

        ArrayAdapter<String> my_adpt1 = new ArrayAdapter<String>(hierc.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.yn));
        my_adpt1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin1.setAdapter(my_adpt1);

        Spinner vicky_spin7 = (Spinner) findViewById(R.id.spinner7);

        ArrayAdapter<String> my_adpt7 = new ArrayAdapter<String>(hierc.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.lk));
        my_adpt7.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin7.setAdapter(my_adpt7);


        Spinner vicky_spin8 = (Spinner) findViewById(R.id.spinner4);

        ArrayAdapter<String> my_adpt8 = new ArrayAdapter<String>(hierc.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.yn));
        my_adpt8.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin8.setAdapter(my_adpt8);



        Spinner vicky_spin3 = (Spinner) findViewById(R.id.spinne2);

        ArrayAdapter<String> my_adpt3 = new ArrayAdapter<String>(hierc.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.yn));
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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.hc;
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
            }, 25000);
        }
    }

    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(hierc.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        X1=(TextView)findViewById(R.id.X1);
        String str1=X1.getText().toString();
        if (!str1.equals("")) {
            gif = findViewById(R.id.gif);
            LinearLayout sc = (LinearLayout) findViewById(R.id.scrollLayout);
            sc.setVisibility(View.GONE);
            gif.setVisibility(View.VISIBLE);
            gif();
            p1 = (TextView) findViewById(R.id.p1);
            ts = (TextView) findViewById(R.id.ts);
            vicky_spin = (Spinner) findViewById(R.id.spinner2);
            vicky_spin1 = (Spinner) findViewById(R.id.spinner7);
            vicky_spin2 = (Spinner) findViewById(R.id.spinner6);
            vicky_spin3 = (Spinner) findViewById(R.id.spinne2);
            vicky_spin4 = (Spinner) findViewById(R.id.spinner4);
            int str4 = vicky_spin.getSelectedItemPosition();
            int str5 = vicky_spin1.getSelectedItemPosition();
            int str15 = vicky_spin2.getSelectedItemPosition();
            int str16 = vicky_spin3.getSelectedItemPosition();
            int str17 = vicky_spin4.getSelectedItemPosition();
            String str6 = ts.getText().toString();
            String str9 = p1.getText().toString();
            pyattr = splash_screen.pyobj.callAttr("hc", str1, str6, str16, str4, str15, str5, str17, str9);
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
            ImageView grp1 = (ImageView) findViewById(R.id.grp1);
            ImageView grp2 = (ImageView) findViewById(R.id.grp2);
            String err = val.get(1).toString();
            new Handler().postDelayed(new Runnable() {
                int k = 0;
                @Override
                public void run() {
                    sc.setVisibility(View.VISIBLE);
                    gif.setVisibility(View.GONE);
                    if (err.equals("error")) {
                        grp1.setVisibility(View.GONE);
                        grp2.setVisibility(View.GONE);
                        sc.getLayoutParams().height = 280;
                        out1.getLayoutParams().height = 190;
                        k = 1;
                    } else {
                        sc.getLayoutParams().height= ViewGroup.LayoutParams.WRAP_CONTENT;
                        final float scale = getResources().getDisplayMetrics().density;
                        int pixels = (int) (236 * scale + 0.5f);
                        out1.getLayoutParams().height = pixels;
                        if (!val.get(1).toString().equals("")) {
                            ViewTreeObserver vto = sc.getViewTreeObserver();
                            vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
                                @Override
                                public void onGlobalLayout() {
                                    sc.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                                    int width = sc.getMeasuredWidth() - 105;
                                    byte data[] = android.util.Base64.decode(val.get(1).toString(), Base64.DEFAULT);
                                    Bitmap bmp = BitmapFactory.decodeByteArray(data, 0, data.length);
                                    grp1.setImageBitmap(bmp);
                                    grp1.getLayoutParams().width = width;
                                    grp1.setVisibility(View.VISIBLE);
                                }
                            });
                        } else {
                            grp1.setVisibility(View.GONE);
                        }
                        if (!val.get(2).toString().equals("")) {
                            ViewTreeObserver vto = sc.getViewTreeObserver();
                            vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
                                @Override
                                public void onGlobalLayout() {
                                    sc.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                                    int width = sc.getMeasuredWidth() - 105;
                                    byte data1[] = android.util.Base64.decode(val.get(2).toString(), Base64.DEFAULT);
                                    Bitmap bmp1 = BitmapFactory.decodeByteArray(data1, 0, data1.length);
                                    grp2.setImageBitmap(bmp1);
                                    grp2.getLayoutParams().width = width;
                                    grp2.setVisibility(View.VISIBLE);
                                }
                            });
                        } else {
                            grp2.setVisibility(View.GONE);
                        }
                    }
                }
            }, 3000);

        }
        else{
            Toast.makeText(this,"Please Enter appropriate data darling",Toast.LENGTH_LONG).show();
        }


    }
}