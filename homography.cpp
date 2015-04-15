#include "homography.h"
#include "matrix.h"

using namespace std;

//helper function
vector<float> applyHomographyPoint(float x, float y, Matrix &H){
  vector<float> out(2,0);
  float xp = H(0,0)*x + H(1,0)*y + H(2,0);
  float yp = H(0,1)*x + H(1,1)*y + H(2,1);
  float wp = H(0,2)*x + H(1,2)*y + H(2,2);
  out[0] = xp/wp;
  out[1] = yp/wp;
  return out;
}

// PS06: apply homography
void applyhomography(const Image &source, Image &out, Matrix &H, bool bilinear) {
  // do something :)
  Matrix Hi = H.inverse();
  for (int x = 0; x < out.width(); x++)
    for (int y = 0; y < out.height(); y++) {
      vector<float> pp = applyHomographyPoint(x, y, Hi);
      float xp = pp[0];
      float yp = pp[1];
      if (xp >= 0 && xp <= source.width() - 1 && yp >= 0 && yp <= source.height() - 1) {
        for (int z = 0; z < out.channels(); z++)
          if (bilinear) {
            out(x, y, z) = interpolateLin(source, xp, yp, z, true);
          }
          else {
            out(x, y, z) = source((int) round(xp), (int) round(yp), z);
          }
      }
    }
}

// PS06: 6.865 or extra credit: apply homography fast
void applyhomographyFast(const Image &source, Image &out, Matrix &H, bool bilinear) {
  // do something :)

  // Hint: a lot of the work is similar to applyhomography(). You may want to
  // write a small helpful function to avoid duplicating code.
}

// PS06: compute homography given a list of point pairs
Matrix computeHomography(const float listOfPairs[4][2][3]) {
  Matrix system(8, 8);
  Matrix vec(1, 8);
  for (int i = 0; i < 4; i++) {
    addConstraint(system, i, listOfPairs[i]);
    vec(0, 2 * i) = listOfPairs[i][1][0];
    vec(0, 2 * i + 1) = listOfPairs[i][1][1];
  }

  Matrix out(3, 3);

  if (system.determinant() != 0){
    Matrix solution = system.inverse().multiply(vec);
    for (int i = 0; i < 8; i++)
      out(i % 3, i / 3) = solution(0, i);
    out(2, 2) = 1;
  }else{
    out(0, 0) = 1;
    out(1, 1) = 1;
    out(2, 2) = 1;
  }

  return out;

}

// PS06: optional function that might help in computeHomography
void addConstraint(Matrix &systm,  int i, const float constr[2][3]) {
    systm(0,2*i) = constr[0][0];
    systm(1,2*i) = constr[0][1];
    systm(2,2*i) = constr[0][2];
    systm(6,2*i) = -constr[1][0]*constr[0][0];
    systm(7,2*i) = -constr[1][0]*constr[0][1];

    systm(3,2*i+1) = constr[0][0];
    systm(4,2*i+1) = constr[0][1];
    systm(5,2*i+1) = constr[0][2];
    systm(6,2*i+1) = -constr[1][1]*constr[0][0];
    systm(7,2*i+1) = -constr[1][1]*constr[0][1];

}

// PS06: compute a transformed bounding box
// returns [xmin xmax ymin ymax]
vector<float> computeTransformedBBox(int imwidth, int imheight, Matrix H) {
  vector <float> minmax(4,0);
  vector<float> upperLeft = applyHomographyPoint(0,0,H);
  vector<float> upperRight = applyHomographyPoint((float)imwidth, 0, H);
  vector<float> bottomLeft = applyHomographyPoint(0, (float)imheight, H);
  vector<float> bottomRight = applyHomographyPoint((float)imwidth, (float)imheight, H);
  minmax[0] = min(upperLeft[0], min(upperRight[0], min(bottomLeft[0], bottomRight[0])));
  minmax[1] = max(upperLeft[0], max(upperRight[0], max(bottomLeft[0], bottomRight[0])));
  minmax[2] = min(upperLeft[1], min(upperRight[1], min(bottomLeft[1], bottomRight[1])));
  minmax[3] = max(upperLeft[1], max(upperRight[1], max(bottomLeft[1], bottomRight[1])));
  return minmax;
}

// PS06: homogenize vector v.
// this function is not required, but would be useful for use in
// computeTransformedBBox()
Matrix homogenize(Matrix &v) {
  Matrix out(1, 3);
  out(0,0) = v(0,0);
  out(0,1) = v(0,1);
  out(0,2) = 1;
}

// PS06: compute a 3x3 translation Matrix
Matrix translate(vector<float> B) {
  Matrix out(3, 3);
  out(0,0)=1;
  out(1,1)=1;
  out(2,2)=1;
  out(2,0)=-B[0];
  out(2,1)=-B[2];
  return out;
}

// PS06: compute the union of two bounding boxes
vector <float> bboxUnion(vector<float> B1, vector<float> B2) {
  vector<float> B(4,0);
  B[0] = min(B1[0], B2[0]);
  B[1] = max(B1[1], B2[1]);
  B[2] = min(B1[2], B2[2]);
  B[3] = max(B1[3], B2[3]);
  return B;
}

// PS06: stitch two images given a list or 4 point pairs
Image stitch(const Image &im1, const Image &im2, const float listOfPairs[4][2][3]) {
  Matrix H = computeHomography(listOfPairs);
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

// PS06: useful for bounding-box debugging.
Image drawBoundingBox(const Image &im, vector<float> minmax) {
  Image output = im;
  float xmin = minmax[0];
  float xmax = minmax[1];
  float ymin = minmax[2];
  float ymax = minmax[3];
  for (int x = xmin; x <= xmax; x++){
    output(x, ymin) = 0;
    output(x, ymax) = 0;
  }
  for (int y = ymin; y <= ymax; y++){
    output(xmin, y) = 0;
    output(xmax, y) = 0;
  }
  return output;
}


/***********************************************************************
 * Do not edit code  below this line
 **********************************************************************/
// get the minimum vector element
float min_vec_elem(vector<float> v) {
  float mmin = FLT_MAX;
  for (int i = 0; i < (int)v.size(); i++) mmin = min(mmin, v[i]);
  return mmin;
}

// get the maximum vector element
float max_vec_elem(vector<float> v) {
  float mmax = -FLT_MAX;
  for (int i = 0; i < (int)v.size(); i++) mmax = max(mmax, v[i]);
  return mmax;
}
