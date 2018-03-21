# latent_walkback

The simplest way of doing it is that we make the transition operator ONLY in hidden variable space (so we parametrize p(next h | previous h) and we also learn a p(x | h) and a q(h | x).

The math is EXACTLY the same if we replace s_0 by x and the other s_i by h_i.

The only difference is that we use a different parametrization for p(s_0 | s_1), i.e., p(x|h), than for p(s_i | s_{i+1)) = p(h_i | h_{i+1)). Similarly there is a different parametrization for q(s_1 | s_0), i.e., q(h|x), than for q(s_{i+1} | s_i) = q(h_{i+1} | h_i). And the dimensions of s_0=x and s_i=h_i don't have to be the same anymore.

It is pretty easy to  implement. In the implementation, we can still visualize the trajectory by sampling p(x | h = s_i)  on intermediate steps to "see" what the latent variable "thinks about" along the way.

This will probably give images that are a bit blurred, like our current VW and like DAEs. However, at some point we can add a GAN discriminator just for the h --> x step and it should get rid of most of the blurriness.

