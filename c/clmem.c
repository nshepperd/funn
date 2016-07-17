#include <CL/cl.h>
#include <stdint.h>

static int64_t count = 0;

void clmem_increase(int64_t x) {
  count += x;
}

void clmem_free(cl_mem obj) {
  size_t size = 0;
  clGetMemObjectInfo(obj, CL_MEM_SIZE, sizeof(size), &size, NULL);
  clReleaseMemObject(obj);
  count -= size;
}

int64_t clmem_count() {
  return count;
}
