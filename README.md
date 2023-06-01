## Projet Python dans le cadre du cours de Proj602_CMI : Restauration d'image.
## Bibliothèques utilisées : numpy et scipy
### Dans ce projet, nous devions effectuer une diffusion anisotropique sur des images. Dans un premier temps, nous avons utilisé la méthode de diffusion de Perona et Malik. Pour ce faire, on diffuse l'image en calculant le gradient en chaque point de l'image.Tout d'abord, il faut calculer les gradients en x et en y. Pour calculer le gradient en x, on fait la soustraction de la valeur en (x+1,y)/2 moins la valeur en (x-1,y)/2. Pour y, c'est la même chose mais avec (x,y+1)/2 et (x,y-1)/2. Pour calculer la diffusion, on applique cette formule :
### g(|∇x|) * ∇x(x+1/2,y) - g(|∇x|) * ∇x(x-1/2,y) + g(|∇y|) * ∇y(x,y+1/2) - g(|∇y|) * ∇y(x,y-1/2)
