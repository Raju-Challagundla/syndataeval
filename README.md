# syndataeval

## tabsyn
<pre>
  conda activate synthcity
</pre>
Use the following command for training:
For Adult Dataset
<pre>
  # train VAE first
  python main.py --dataname adult --method vae --mode train
  
  # after the VAE is trained, train the diffusion model
  python main.py --dataname adult --method tabsyn --mode train
</pre>
 Use the following command for training: For For Credit Risk Dataset
<pre>
  # train VAE first
  python main.py --dataname credit --method vae --mode train
  
  # after the VAE is trained, train the diffusion model
  python main.py --dataname credit --method tabsyn --mode train
</pre>
<pre>
  # evaluation metrics
   python eval/eval_quality_imputed.py --dataname credit_risk_dataset --model tabsyn --path synthetic/credit_risk_dataset/tabsyn.csv
   python eval/eval_quality_imputed.py --dataname adult --model tabsyn --path synthetic/adult/tabsyn.csv

</pre>
