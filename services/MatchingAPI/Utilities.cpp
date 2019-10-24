#include "Utilities.h"

Mat getMat(string imageBytes)
{
    size_t length = imageBytes.size();
    Mat imageMat;
    vector<char> data((char*)imageBytes.c_str(), (char*)imageBytes.c_str() + length);
    imageMat = imdecode(data, IMREAD_COLOR);
    return imageMat;
}

string convertImgToString (Mat img)
{
    vector<uchar> buf;
    imencode(".jpg", img, buf );
    auto *enc_msg = new uchar[buf.size()];
    for(int i=0; i < buf.size(); i++) enc_msg[i] = buf[i];
    string encoded = base64_encode(enc_msg, buf.size());
    return encoded;
}


string getImageString(string path)
{
    FILE *in_file  = fopen(path.c_str(), "rb");

    fseek(in_file, 0L, SEEK_END);
    int sz = ftell(in_file);
    rewind(in_file);
    char imageBytes[sz];
    fread(imageBytes, sizeof *imageBytes, sz, in_file);
    string image_bytes(imageBytes, sz);
    fclose(in_file);
    return image_bytes;
}