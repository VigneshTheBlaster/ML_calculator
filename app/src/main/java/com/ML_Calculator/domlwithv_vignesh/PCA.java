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

public class PCA extends AppCompatActivity {

    PyObject pyattr;
    TextView X1,ts;
    ImageView gif,demo;
    VideoView vv;
    int pls=0;
    Spinner vicky_spin;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pca2);
        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
        TextView tb=(TextView)findViewById(R.id.textView);
        tb.setText("P  C  A(analysis)");
        demo=(ImageView)findViewById(R.id.mutton);
        demo.setBackgroundResource(R.drawable.play_icon);
        Spinner vicky_spin = (Spinner) findViewById(R.id.spinner3);

        ArrayAdapter<String> my_adpt = new ArrayAdapter<String>(PCA.this, R.layout.vicky_spinner,getResources().getStringArray(R.array.yn));
        my_adpt.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        vicky_spin.setAdapter(my_adpt);
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
            String str = "android.resource://" + getPackageName() + "/" + R.raw.pca;
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
        Glide.with(PCA.this).asGif().load(R.raw.finalgif).into(gif);
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
            ts = (TextView) findViewById(R.id.rs);
            vicky_spin = (Spinner) findViewById(R.id.spinner3);
            int str4 = vicky_spin.getSelectedItemPosition();
            String str6 = ts.getText().toString();
            pyattr = splash_screen.pyobj.callAttr("pca", str1, str4, str6);
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
                @Override
                public void run() {
                    sc.setVisibility(View.VISIBLE);
                    gif.setVisibility(View.GONE);
                    if (err.equals("error")) {
                        grp1.setVisibility(View.GONE);
                        sc.getLayoutParams().height = 280;
                        out1.getLayoutParams().height = 190;
                    } else {
                        if (!err.equals("")) {
                            sc.getLayoutParams().height= ViewGroup.LayoutParams.WRAP_CONTENT;
                            final float scale = getResources().getDisplayMetrics().density;
                            int pixels = (int) (236 * scale + 0.5f);
                            out1.getLayoutParams().height = pixels;
                            grp1.setVisibility(View.VISIBLE);
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
                                }
                            });
                        } else {
                            grp1.setVisibility(View.GONE);
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