#include "blending.h"
#include "matrix.h"
#include <math.h>
#include <ctime>

using namespace std;

/*****************************************************************************
 * blending related functions re-written from previous asasignments
 *****************************************************************************/

// PSet 08 instead of writing source in out, *add* the source to out based on the weight
// so out(x,y) = out(x, y) + weight * image
void applyhomographyBlend(const Image &source, const Image &weight, Image &out, Matrix &H, bool bilinear) {
  Matrix Hi = H.inverse();
  for (int x = 0; x < out.width(); x++)
    for (int y = 0; y < out.height(); y++) {
      vector<float> pp = applyHomographyPoint(x, y, Hi);
      float xp = pp[0];
      float yp = pp[1];
      if (xp >= 0 && xp <= source.width() - 1 && yp >= 0 && yp <= source.height() - 1) {
        for (int z = 0; z < out.channels(); z++)
          if (bilinear) {
            out(x, y, z) += interpolateLin(source, xp, yp, z) * interpolateLin(weight, xp, yp, 0);
          }
          else {
            out(x, y, z) += source((int) round(xp), (int) round(yp), z) * weight((int) round(xp), (int) round(yp), 0);
          }
      }
    }
}

// PSet 08 stitch using image weights.
// note there is no weight normalization.
Image stitchLinearBlending(const Image &im1, const Image &im2, const Image &we1, const Image &we2, Matrix H) {
  vector<float> bbox1 = computeTransformedBBox(im1.width(), im1.height(), H);
  Matrix T1 = translate(bbox1);
  vector<float> bbox2 = vector<float>(4,0);
  bbox2[1] = im2.width();
  bbox2[3] = im2.height();
  vector<float> bbox = bboxUnion(bbox1, bbox2);
  Image im(bbox[1]-bbox[0], bbox[3]-bbox[2],3);
  Matrix H1 = T1.multiply(H);
  applyhomographyBlend(im2, we2, im, T1, true);
  applyhomographyBlend(im1, we1, im, H1, true);
  return im;
}

Image stitch(const Image &im1, const Image &im2, Matrix H){
  vector<float> bbox1 = computeTransformedBBox(im1.width(), im1.height(), H);
  Matrix T1 = translate(bbox1);
  vector<float> bbox2 = vector<float>(4,0);
  bbox2[1] = im2.width();
  bbox2[3] = im2.height();
  vector<float> bbox = bboxUnion(bbox1, bbox2);
  Image im(bbox[1]-bbox[0], bbox[3]-bbox[2],3);
  Matrix H1 = T1.multiply(H);
  applyhomography(im2, im, T1, true);
  applyhomography(im1, im, H1, true);
  return im;
}
/*****************************************************************************
 * blending functions Pset 08
 *****************************************************************************/

// PSet 08 weight image
Image blendingweight(int imwidth, int imheight) {
  Image weights(imwidth, imheight, 1);
  float center_x = (imwidth-1)/2;
  float center_y = (imheight-1)/2;
  for(int x = 0; x < imwidth; x++) {
    float xweight = 1 - fabs(center_x - x) / center_x;
    for (int y = 0; y < imheight; y++) {
      float yweight = 1 - fabs(center_y - y) / center_y;
      weights(x, y) = xweight * yweight;
    }
  }
  return weights;
}

// Optional: low freq and high freq (2-scale) decomposition
vector<Image> scaledecomp(const Image &im, float sigma) {
  vector <Image> ims;
  Image lowpass = gaussianBlur_seperable(im, sigma);
  Image highpass = im - lowpass;
  ims.push_back(lowpass);
  ims.push_back(highpass);
  return ims;
}

// PSet 08 stitch using different blending models
// blend can be 0 (none), 1 (linear) or 2 (2-layer)
// use more helper functions if necessary
Image stitchBlending(Image &im1, Image &im2, Matrix H, int blend) {
  switch(blend){
    case 0: {
      return stitch(im1, im2, H);
    }
    case 1: {
      Image w1 = blendingweight(im1.width(), im1.height());
      Image w2 = blendingweight(im2.width(), im2.height());


      Image white1(im1.width(), im1.height(), im1.channels());
      Image white2(im2.width(), im2.height(), im2.channels());
      white1 = white1 + 1;
      white2 = white2 + 1;

      vector<float> bbox1 = computeTransformedBBox(im1.width(), im1.height(), H);
      Matrix T1 = translate(bbox1);
      vector<float> bbox2 = vector<float>(4,0);
      bbox2[1] = im2.width();
      bbox2[3] = im2.height();
      vector<float> bbox = bboxUnion(bbox1, bbox2);
      Image im(bbox[1]-bbox[0], bbox[3]-bbox[2],3);
      Image sumOfWeights(im.width(), im.height(), 1);
      Matrix H1 = T1.multiply(H);
      applyhomographyBlend(im2, w2, im, T1, true);
      applyhomographyBlend(im1, w1, im, H1, true);

      applyhomographyBlend(white2, w2, sumOfWeights, T1, true);
      applyhomographyBlend(white1, w1, sumOfWeights, H1, true);
      return im / (copychannels(sumOfWeights, im1.channels()) + 0.000000001);
    }
    case 2:
      return Image(1,1,1);
  }
}

// PSet 08 auto stitch
Image autostitch(Image &im1, Image &im2, int blend, float blurDescriptor, float radiusDescriptor) {
  vector<Point> corners1 = HarrisCorners(im1);
  vector<Point> corners2 = HarrisCorners(im2);
  vector<Feature> features1 = computeFeatures(im1, corners1, blurDescriptor, radiusDescriptor);
  vector<Feature> features2 = computeFeatures(im2, corners2, blurDescriptor, radiusDescriptor);

  vector<Correspondance> corr = findCorrespondences(features1, features2);
  Matrix H = RANSAC(corr);

  return stitchBlending(im1, im2, H, blend);
}

/************************************************************************
 * Coolness: mini planets.
 ************************************************************************/

Image pano2planet(const Image &pano, int newImSize, bool clamp) {
  return Image(0);
}


/************************************************************************
 * 6.865: Stitch N images into a panorama
 ************************************************************************/

// Pset08-865. Compute sequence of N-1 homographies going from Im_i to Im_{i+1}
// Implement me!
vector<Matrix> sequenceHs(vector<Image> ims, float blurDescriptor, float radiusDescriptor) {
  vector<Matrix> Hs;
  return Hs;
}

// Pset08-865. stack homographies:
//   transform a list of (N-1) homographies that go from I_i to I_i+1
//   to a list of N homographies going from I_i to I_refIndex.
vector <Matrix> stackHomographies(vector <Matrix> Hs, int refIndex) {
  vector <Matrix> gHs;
  return gHs;
}

// Pset08-865: compute bbox around N images given one main reference.
vector<float> bboxN(const vector<Matrix> &Hs, const vector<Image> &ims) {
  vector<float> ubbox;
  return ubbox;
}

// Pset08-865.
// Implement me!
Image autostitchN(vector<Image> ims, int refIndex, float blurDescriptor, float radiusDescriptor) {
  return Image(0);
}


/******************************************************************************
 * Helpful functions
 *****************************************************************************/

 // copy a single-channeled image to several channels
Image copychannels(const Image &im, int nChannels) {
 assert(im.channels() == 1, "image must have one channel");
 Image oim(im.width(), im.height(), nChannels);

 for (int i = 0; i < im.width(); i++) {
   for (int j = 0; j < im.height(); j++) {
     for (int c = 0; c < nChannels; c++) {
       oim(i, j, c) = im(i, j);
     }
   }
 }
 return oim;
}

// create an n x n identity matrix
Matrix eye(int n) {
  Matrix m(n, n);
  for (int i = 0; i < n; i++) m(i, i) = 1;
  return m;
}
