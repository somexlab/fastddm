// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_MEMCHK_GPU_H__
#define __HELPER_MEMCHK_GPU_H__

/*! \file helper_memchk_gpu.h
    \brief Declaration of C++ helper functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include <vector>
#include <string>

#include <cufft.h>

using namespace std;

// *** code ***

/*! \brief Get free host memory (in bytes)
    \param free_mem     Host free memory returned (in bytes)
*/
void get_host_free_mem(unsigned long long &free_mem);

/*! \brief Estimate and check host memory needed for diff mode
    \param nx           Number of fft nodes, x direction
    \param ny           Number of fft nodes, y direction
    \param length       Number of frames
    \param lags         Vector of lags to analyze
 */
void chk_host_mem_diff(unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long length,
                       vector<unsigned int> lags);

/*! \brief Estimate and check host memory needed for fft mode
    \param nx           Number of fft nodes, x direction
    \param ny           Number of fft nodes, y direction
    \param length       Number of frames
 */
void chk_host_mem_fft(unsigned long long nx,
                      unsigned long long ny,
                      unsigned long long length);

/*! \brief Get free device memory (in bytes)
    \param free_mem     Device free memory returned (in bytes)
*/
void get_device_free_mem(unsigned long long &free_mem);

/*! \brief Get the device memory pitch for multiple arrays of length N
    \param N        subarray size
    \param Nbytes   element size in bytes
 */
unsigned long long get_device_pitch(unsigned long long N,
                                    int Nbytes);

/*! \brief Get the device memory for fft2
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param nt           number of elements (in t direction)
    \param cufft_res    result of cufft function
 */
unsigned long long get_device_fft2_mem(unsigned long long nx,
                                       unsigned long long ny,
                                       unsigned long long nt,
                                       cufftResult &cufft_res);

/*! \brief Get the device memory for fft
    \param nt       number of fft nodes in t direction
    \param N        number of elements
    \param pitch    pitch of input array
    \param cufft_res    result of cufft function
 */
unsigned long long get_device_fft_mem(unsigned long long nt,
                                      unsigned long long N,
                                      unsigned long long pitch,
                                      cufftResult &cufft_res);

/*! \brief Optimize fft2 execution parameters based on available gpu memory
    \param pitch_buff       Pitch of buffer device array
    \param num_fft2         Number of fft2 batches
    \param is_input_double  True if pixel value is double
    \param pixel_Nbytes     Number of bytes per pixel
    \param width            Width of the image
    \param height           Height of the image
    \param length           Number of frames
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param free_mem         Available gpu memory
 */
void optimize_fft2(unsigned long long &pitch_buff,
                   unsigned long long &num_fft2,
                   bool is_input_double,
                   unsigned long long pixel_Nbytes,
                   unsigned long long width,
                   unsigned long long height,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long free_mem);

/*! \brief Optimize fullshift execution parameters based on available gpu memory
    \param pitch_fs         Pitch of device array for full and shift operation
    \param num_fullshift    Number of full and shift chunks
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param num_lags         Number of lags analysed
    \param free_mem         Available gpu memory
 */
void optimize_fullshift(unsigned long long &pitch_fs,
                        unsigned long long &num_fullshift,
                        unsigned long long nx,
                        unsigned long long ny,
                        unsigned long long num_lags,
                        unsigned long long free_mem);

/*! \brief Optimize structure function diff execution parameters based on available gpu memory
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param num_chunks       Number of q points chunks
    \param length           Number of frames
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param num_lags         Number of lags analysed
    \param free_mem         Available gpu memory
 */
void optimize_diff(unsigned long long &pitch_q,
                   unsigned long long &pitch_t,
                   unsigned long long &num_chunks,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long num_lags,
                   unsigned long long free_mem);

/*! \brief Optimize structure function fft execution parameters based on available gpu memory
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param pitch_nt         Pitch of workspace1 device array (nt-pitch)
    \param num_chunks       Number of q points chunks
    \param length           Number of frames
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param nt               Number of fft nodes, t direction
    \param num_lags         Number of lags analysed
    \param free_mem         Available gpu memory
 */
void optimize_fft(unsigned long long &pitch_q,
                  unsigned long long &pitch_t,
                  unsigned long long &pitch_nt,
                  unsigned long long &num_chunks,
                  unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long nt,
                  unsigned long long num_lags,
                  unsigned long long free_mem);

/*! \brief Estimate device memory needed for diff mode and optimize
    \param width            Width of the image
    \param height           Height of the image
    \param pixel_Nbytes     Number of bytes per pixel
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param length           Number of frames
    \param lags             Vector of lags to analyze
    \param is_input_double  True if pixel value is double
    \param num_fft2         Number of fft2 batches
    \param num_chunks       Number of q points chunks
    \param num_fullshift    Number of full and shift chunks
    \param pitch_buff       Pitch of buffer device array
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param pitch_fs         Pitch of device array for full and shift operation
 */
void chk_device_mem_diff(unsigned long long width,
                         unsigned long long height,
                         int pixel_Nbytes,
                         unsigned long long nx,
                         unsigned long long ny,
                         unsigned long long length,
                         vector<unsigned int> lags,
                         bool is_input_double,
                         unsigned long long &num_fft2,
                         unsigned long long &num_chunks,
                         unsigned long long &num_fullshift,
                         unsigned long long &pitch_buff,
                         unsigned long long &pitch_q,
                         unsigned long long &pitch_t,
                         unsigned long long &pitch_fs);

/*! \brief Estimate device memory needed for fft mode and optimize
    \param width            Width of the image
    \param height           Height of the image
    \param pixel_Nbytes     Number of bytes per pixel
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param nt               Number of fft nodes, t direction
    \param length           Number of frames
    \param lags             Vector of lags to analyze
    \param is_input_double  True if pixel value is double
    \param num_fft2         Number of fft2 batches
    \param num_chunks       Number of q points chunks
    \param num_fullshift    Number of full and shift chunks
    \param pitch_buff       Pitch of buffer device array
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param pitch_nt         Pitch of workspace1 device array (nt-pitch)
    \param pitch_fs         Pitch of device array for full and shift operation
 */
void chk_device_mem_fft(unsigned long long width,
                        unsigned long long height,
                        int pixel_Nbytes,
                        unsigned long long nx,
                        unsigned long long ny,
                        unsigned long long nt,
                        unsigned long long length,
                        vector<unsigned int> lags,
                        bool is_input_double,
                        unsigned long long &num_fft2,
                        unsigned long long &num_chunks,
                        unsigned long long &num_fullshift,
                        unsigned long long &pitch_buff,
                        unsigned long long &pitch_q,
                        unsigned long long &pitch_t,
                        unsigned long long &pitch_nt,
                        unsigned long long &pitch_fs);

#endif