#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define CSC(call) 														                        \
while (1) 																				        \
{																							    \
	cudaError_t status = call;									                                \
	if (status != cudaSuccess) {								                                \
		printf("ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
		exit(0);																	            \
	}																						    \
	break;																					    \
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    for(y = idy; y < h; y += offset_y)
        for(x = idx; x < w; x += offset_x) {
            float Gx = 0.0;
            float Gy = 0.0;

            y = max(min(y, h), 0);
            x = max(min(x, w), 0);

            //horizontal
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx += 2) {
                    p = tex2D<uchar4>(tex, x + kx, y + ky);
                    float Y = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
                    Gx += kx * Y;
                }
            }

            // vertical
            for (int ky = -1; ky <= 1; ky += 2) {
                for (int kx = -1; kx <= 1; kx++) {
                    p = tex2D<uchar4>(tex, x + kx, y + ky);
                    float Y = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
                    Gy += ky * Y;
                }
            }
        
            float gradient = sqrt(Gx * Gx + Gy * Gy);
            gradient = min(max(gradient, 0.0f), 255.0f);

            out[y * w + x] = make_uchar4(gradient, gradient, gradient, p.w);
        }
}

int main() 
{
    char inputFile[255], outputFile[255];
    scanf("%s", inputFile);
    scanf("%s", outputFile);

    int width, height;
    FILE *fp = fopen(inputFile, "rb");
    fread(&width, sizeof(int), 1, fp);
    fread(&height, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * width * height);
    fread(data, sizeof(uchar4), width * height, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, width, height));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * width * height));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, width, height);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(outputFile, "wb");
    fwrite(&width, sizeof(int), 1, fp);
    fwrite(&height, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), width * height, fp);
    fclose(fp);

    free(data);
    return 0;
}