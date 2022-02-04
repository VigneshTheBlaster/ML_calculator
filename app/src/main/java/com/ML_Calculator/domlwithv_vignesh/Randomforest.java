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

public class Randomforest extends AppCompatActivity {

    PyObject pyattr;
    TextView X1,p1,ts,rs,msl,mln,ccv,sr,st,pre,not,mf;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin,vicky_spin2,vicky_spin4,vicky_spin5;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_randomforest);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("Random Forest");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);

        Spinner vicky_spin = (Spinner) findViewById(R.id.spinner3);
        ArrayAdapter<String> my_adpt = new ArrayAdapter<String>(Randomforest.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.testing_method));
        my_adpt.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin.setAdapter(my_adpt);

        Spinner vicky_spin1 = (Spinner) findViewById(R.id.spinne6);
        ArrayAdapter<String> my_adpt1 = new ArrayAdapter<String>(Randomforest.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.sty));
        my_adpt1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin1.setAdapter(my_adpt1);

        Spinner vicky_spin2 = (Spinner) findViewById(R.id.spinne66);
        ArrayAdapter<String> my_adpt2 = new ArrayAdapter<String>(Randomforest.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.ct));
        my_adpt2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        vicky_spin2.setAdapter(my_adpt2);

        Spinner vicky_spin3 = (Spinner) findViewById(R.id.spinne77);
        ArrayAdapter<String> my_adpt3 = new ArrayAdapter<String>(Randomforest.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.sc));
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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.rf;
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
            }, 51000);
        }
    }


    public void gif(){
        InputMethodManager input=(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        input.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(),InputMethodManager.HIDE_NOT_ALWAYS);
        gif=findViewById(R.id.gif);
        Glide.with(Randomforest.this).asGif().load(R.raw.finalgif).into(gif);
    }

    public void build1(View view){
        X1=(TextView)findViewById(R.id.fn);
        String str1=X1.getText().toString();
        if (!str1.equals("")) {
            gif = findViewById(R.id.gif);
            LinearLayout sc = (LinearLayout) findViewById(R.id.scrollLayout);
            sc.setVisibility(View.GONE);
            gif.setVisibility(View.VISIBLE);
            gif();
            p1 = (TextView) findViewById(R.id.X1);
            ts = (TextView) findViewById(R.id.y1);
            vicky_spin = (Spinner) findViewById(R.id.spinne66);
            vicky_spin2 = (Spinner) findViewById(R.id.spinner3);
            vicky_spin4 = (Spinner) findViewById(R.id.spinne6);
            vicky_spin5 = (Spinner) findViewById(R.id.spinne77);
            rs = (TextView) findViewById(R.id.lp1);
            msl = (TextView) findViewById(R.id.md);
            mln = (TextView) findViewById(R.id.mln);
            ccv = (TextView) findViewById(R.id.cv);
            sr = (TextView) findViewById(R.id.rs);
            st = (TextView) findViewById(R.id.ts);
            pre = (TextView) findViewById(R.id.p);
            not = (TextView) findViewById(R.id.not);
            mf = (TextView) findViewById(R.id.mf);

            String str9 = p1.getText().toString();
            String str6 = ts.getText().toString();
            int str4 = vicky_spin.getSelectedItemPosition();
            int str17 = vicky_spin2.getSelectedItemPosition();
            int str16 = vicky_spin4.getSelectedItemPosition();
            int str18 = vicky_spin5.getSelectedItemPosition();
            String str5 = rs.getText().toString();
            String str7 = msl.getText().toString();
            String str8 = mln.getText().toString();
            String str10 = ccv.getText().toString();
            String str11 = sr.getText().toString();
            String str12 = st.getText().toString();
            String str13 = pre.getText().toString();
            String str14 = not.getText().toString();
            String str15 = mf.getText().toString();

            pyattr = splash_screen.pyobj.callAttr("rf", str1, str9, str6, str4, str17, str16, str18, str14, str15, str5, str7, str8, str10, str11, str12, str13);

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
            String err = val.get(1).toString();
            new Handler().postDelayed(new Runnable() {
                int k = 0;

                @Override
                public void run() {
                    if (err.equals("error")) {
                        grp1.setVisibility(View.GONE);
                        sc.getLayoutParams().height = 280;
                        out1.getLayoutParams().height = 190;
                        k = 1;
                    } else {
                        sc.getLayoutParams().height= ViewGroup.LayoutParams.WRAP_CONTENT;
                        final float scale = getResources().getDisplayMetrics().density;
                        int pixels = (int) (236 * scale + 0.5f);
                        out1.getLayoutParams().height = pixels;
                        grp1.setVisibility(View.VISIBLE);
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
                                    grp1.setVisibility(View.VISIBLE);
                                    grp1.getLayoutParams().width = width;
                                }
                            });
                        } else {
                            grp1.setVisibility(View.GONE);
                        }
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