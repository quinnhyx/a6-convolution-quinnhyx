[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CmLQfDS4)
# A6. Convolution

Due Date: 11/25 at midnight (optional)

## Instructions

Implement two versions of the one-dimensional (1D) convolution kernel.
Use 0's for boundary/missing values/elements.

1. Naive implementation (10.2 slide 10)
2. Utilize constant cache memory for mask (10.2 slides 22 - 26)

Compare the performance of these two approaches.

## Reflection Questions

1. When does it make sense to use constant memory?
2. What went well with this assignment?
3. What was difficult?
4. How would you approach differently?
5. Anything else you want me to know?

## Submission

- [ ] Convolution Code Versions
  - [ ] Naive 1D w/ 0 boundaries
  - [ ] Mask in constant cache memory
- [ ] Reflection

# Debugging

If you encounter errors, I recommend two approahces:

1. Adding print statements both inside your kernel function
   as well as outside.
   This can include: `printf("%s\n", cudaGetErrorString(cudaGetLastError()));`
   to catch any cuda errors.

2. You can check to make sure that you code is not accessing unallocated memory
   by utilizing NVIDIA's memory sanitizer tool.
   You can run it ok `keroppi` using the following line.
   ```sh
   compute-sanitizer --tool memcheck ./your_cuda_executable_not_source
   ```

# Updates

To update this assignment as changes are made,
a new PR will be generated.
You can find the tab [here](../../pulls).
On that page you can merge the pull request to get the update instructions.
This may invovle rebasing or merging your contributions, reach out
if you need help with this.

## Extras

1. Utilize different boundary conditions
  - Replicate the edge values
  - Utilize identity values (e.g. 1 for multiply)
2. Compare the performance with a CPU based convolution
3. Implement the 2D convolution kernel
4. Implement the tiled version of convoluion (10.2 bonus slides)
