A very simple and vibe-coded point-cloud similarity score, based loosely on Steve Knox's presentation "Extending pairwise element similarity to set similarity efficiently".

I'm a little bit sceptical of this actually being a good score, for a few reasons:

(1) The maximum value of R that we integrate over is very big; surely better to have something that focuses more on typical distances.  

(2) Most of the time, this focuses on whether a new point is far from everyone (unless it happens to form a bridge). It might make more sense to count more robust notions of bridginess, so that you get a big bonus for connecting two huge clusters but a much smaller bonus for connecting a big cluster to a point.  

(3) Add your smarter comments here.
