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
