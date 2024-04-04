# Dilation_and_Erosion_with_Median_Filter
## Author: Neha Prakash ##

---

## How to Run code ? ##

- **Step 1** : **MAKE SURE THAT YOU ARE RUNNING A *LINUX MACHINE* :)**

- **Step 2** : That's all !!! Windows Sucks!!

- **Step 3** : Follow following steps, (I like unnecessary tautologies)
    - Read the `main.cpp` thoroughly and change the *`median_kernel_size`* to the value that you want
        ```C++
        int median_kernel_size = 3;
        ```
        **It's on line 122**.

    - For Dilation and Erosion change the morphological kernel to the pattern that you want. We are very smart, hence we have given you enough freedom to choose your own kernel pattern. Choose wisely!!!
        ```C++
            //////////////// Do calculation on the host side : declare the structuring element
            int struc_ele_width = 5;
            int struc_ele_height = 5;
            int struc_element_count = struc_ele_width*struc_ele_height;
            std::vector<int> dilation_element(5*5);
            memset(dilation_element.data(), 0, struc_element_count*sizeof(int ));
            //    dilation_element = {
            //            0, 0, 1, 1, 1, 0, 0,
            //            0, 0, 1, 1, 1, 0, 0,
            //            1, 1, 1, 1, 1, 1, 1,
            //            1, 1, 1, 1, 1, 1, 1,
            //            1, 1, 1, 1, 1, 1, 1,
            //            0, 0, 1, 1, 1, 0, 0,
            //            0, 0, 1, 1, 1, 0, 0};

            dilation_element = {
                    0, 0, 1, 0, 0,
                    0, 1, 1, 1, 0,
                    1, 1, 1, 1, 1,
                    0, 1, 1, 1, 0,
                    0, 0, 1, 0, 0};

            //int dilation_element[3][3] = {
            //        {0, 1, 0},
            //        {1, 1, 1},
            //        {0, 1, 0}
            //};
            structuring_element a_struc_ele = {
                    dilation_element.data(),
                    struc_ele_width,
                    struc_ele_height,
                    3,                          //  Candidate pixel coordinates
                    3                           //  Candidate pixel coordinates
            };

        ```

    - Now you are all set. To give us 1 GPA in **High Performance Computing with Graphics Card** Course!

    - Change Directory to you favorite directory with : `cd /path/to/your/favorite/directory`

    - CMAKE : `cmake /path/to/CMakeLists.txt`

    - MAKE : `make` (for multiple cores it should be `make -jN` where N is number of cores)

    - Execute the executable generated with following arguments in the same order of inputs : 
        ```bash
        ./GPU_Team2_Project /path/to/your/favorite/image.pgm /path/to/store/the/outputs/
        ```

---
## NOTES: ##
- For your ease we have included a folder with Images [`/path-of-project/Images/Inputs`] in it. 
    - The images are segregated in two directories `Binary` and `SNP_Noise`. These are nothing but Binary Images and Images with salt and pepper noise. 
    - Note that the Median Filter output makes more sense only with Salt and Pepper Noise
    - The output for Dilation, Erosion and Opening, Closing will be a black image if you input a Salt and Pepper image, It is specifically meant to be ran on Binary Images. 

- ENJOY !!! Because you only live once. And you will probably give grade to us only once so why not to make it a 1. :)
