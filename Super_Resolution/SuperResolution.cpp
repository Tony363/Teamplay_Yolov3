#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

int main(int argc, char *argv[])
{
    //Create the module's object
    DnnSuperResImpl sr;

    //Set the image you would like to upscale
    string img_path = "image.png";
    Mat img = cv::imread(img_path);

    //Read the desired model
    string path = "FSRCNN_x2.pb";
    sr.readModel(path);

    //Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 2);

    //Upscale
    Mat img_new;
    sr.upsample(img, img_new);
    cv::imwrite( "upscaled.png", img_new);

    return 0;
}