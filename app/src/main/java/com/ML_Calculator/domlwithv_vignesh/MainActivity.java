package com.ML_Calculator.domlwithv_vignesh;


import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import com.google.android.gms.ads.AdRequest;
import com.google.android.gms.ads.AdView;


public class MainActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        AdView adView = findViewById(R.id.adView);
        AdRequest adRequest=new AdRequest.Builder().build();
        adView.loadAd(adRequest);
    }


    public void ply(View v){
        try {
            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("text/plain");
            shareIntent.putExtra(Intent.EXTRA_SUBJECT, "ML Calculator");
            String shareMessage= "\nHi, this is v.vignesh Let me recommend you first machine learning calculator across internet.\n\n";
            shareMessage = shareMessage + "https://play.google.com/store/apps/details?id=" + BuildConfig.APPLICATION_ID +"\n\n";
            shareIntent.putExtra(Intent.EXTRA_TEXT, shareMessage);
            startActivity(Intent.createChooser(shareIntent, "choose one"));
        } catch(Exception e) {
            //e.toString();
        }
    }


    public void linear(View v){
        Intent myIntent = new Intent(this,linear.class);
        startActivity(myIntent);
    }


    public void plinear(View v){
        Intent myIntent = new Intent(this,polyres.class);
        startActivity(myIntent);
    }

    public void mlinear(View v){
        Intent myIntent = new Intent(this,Multilin.class);
        startActivity(myIntent);
    }

    public void llinear(View v){
        Intent myIntent = new Intent(this,logistic.class);
        startActivity(myIntent);
    }

    public void knn(View v){
        Intent myIntent = new Intent(this,Knearest.class);
        startActivity(myIntent);
    }

    public void svml(View v){
        Intent myIntent = new Intent(this,svmL.class);
        startActivity(myIntent);
    }

    public void svmk(View v){
        Intent myIntent = new Intent(this,svmk.class);
        startActivity(myIntent);
    }

    public void nbs(View v){
        Intent myIntent = new Intent(this,NaiveBayes_str.class);
        startActivity(myIntent);
    }

    public void nbi(View v){
        Intent myIntent = new Intent(this,NaiveBayes_int.class);
        startActivity(myIntent);
    }

    public void km(View v){
        Intent myIntent = new Intent(this,kmeans.class);
        startActivity(myIntent);
    }

    public void hc(View v){
        Intent myIntent = new Intent(this,hierc.class);
        startActivity(myIntent);
    }

    public void dc(View v){
        Intent myIntent = new Intent(this,densityc.class);
        startActivity(myIntent);
    }

    public void fc(View v){
        Intent myIntent = new Intent(this,fuzzyc.class);
        startActivity(myIntent);
    }

    public void zt(View v){
        Intent myIntent = new Intent(this,ztest.class);
        startActivity(myIntent);
    }

    public void tt(View v){
        Intent myIntent = new Intent(this,ttest.class);
        startActivity(myIntent);
    }

    public void ft(View v){
        Intent myIntent = new Intent(this,ftest.class);
        startActivity(myIntent);
    }

    public void ct(View v){
        Intent myIntent = new Intent(this,chitest.class);
        startActivity(myIntent);
    }

    public void dt(View v){
        Intent myIntent = new Intent(this,decisiontree.class);
        startActivity(myIntent);
    }

    public void rf(View v){
        Intent myIntent = new Intent(this,Randomforest.class);
        startActivity(myIntent);
    }

    public void pca(View v){
        Intent myIntent = new Intent(this,PCA.class);
        startActivity(myIntent);
    }

    public void kpca(View v){
        Intent myIntent = new Intent(this,kpca.class);
        startActivity(myIntent);
    }

}