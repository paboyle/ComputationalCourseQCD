#include <vector>
#include <iostream>
#include <stdint.h>
#include <mpi.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>

#include <sys/time.h>
inline double usecond(void) {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1.0*tv.tv_usec + 1.0e6*tv.tv_sec;
}

#define GRID_SYCL_LEVEL_ZERO_IPC

cl::sycl::queue *theGridAccelerator;
cl::sycl::queue *tmpGridAccelerator;

uint32_t acceleratorThreads(void);   
void     acceleratorThreads(uint32_t);
void     acceleratorInit(void);

#define accelerator 
#define accelerator_inline strong_inline

#define accelerator_for2dNB( iter1, num1, iter2, num2, nsimd, ... )	\
  theGridAccelerator->submit([&](cl::sycl::handler &cgh) {		\
      unsigned long nt=acceleratorThreads();				\
      unsigned long unum1 = num1;					\
      unsigned long unum2 = num2;					\
      if(nt < 8)nt=8;							\
      cl::sycl::range<3> local {nt,1,nsimd};				\
      cl::sycl::range<3> global{unum1,unum2,nsimd};			\
      cgh.parallel_for(					\
      cl::sycl::nd_range<3>(global,local), \
      [=] (cl::sycl::nd_item<3> item) /*mutable*/     \
      [[intel::reqd_sub_group_size(8)]]	      \
      {						      \
      auto iter1    = item.get_global_id(0);	      \
      auto iter2    = item.get_global_id(1);	      \
      auto lane     = item.get_global_id(2);	      \
      { __VA_ARGS__ };				      \
     });	   			              \
    });

#define accelerator_barrier(dummy) theGridAccelerator->wait();

#define accelerator_forNB( iter1, num1, nsimd, ... ) accelerator_for2dNB( iter1, num1, iter2, 1, nsimd, {__VA_ARGS__} );

#define accelerator_for( iter, num, nsimd, ... )		\
  accelerator_forNB(iter, num, nsimd, { __VA_ARGS__ } );	\
  accelerator_barrier(dummy);

#define accelerator_for2d(iter1, num1, iter2, num2, nsimd, ... )	\
  accelerator_for2dNB(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

inline void *acceleratorAllocShared(size_t bytes){ return malloc_shared(bytes,*theGridAccelerator);};
inline void *acceleratorAllocDevice(size_t bytes){ return malloc_device(bytes,*theGridAccelerator);};
inline void acceleratorFreeShared(void *ptr){free(ptr,*theGridAccelerator);};
inline void acceleratorFreeDevice(void *ptr){free(ptr,*theGridAccelerator);};
inline void acceleratorCopyDeviceToDevice(void *from,void *to,size_t bytes)  { theGridAccelerator->memcpy(to,from,bytes); theGridAccelerator->wait();}
inline void acceleratorCopyToDevice(void *from,void *to,size_t bytes)  { theGridAccelerator->memcpy(to,from,bytes); theGridAccelerator->wait();}
inline void acceleratorCopyFromDevice(void *from,void *to,size_t bytes){ theGridAccelerator->memcpy(to,from,bytes); theGridAccelerator->wait();}
inline void acceleratorMemSet(void *base,int value,size_t bytes) { theGridAccelerator->memset(base,value,bytes); theGridAccelerator->wait();}

int      acceleratorAbortOnGpuError=1;
uint32_t accelerator_threads=2;
uint32_t acceleratorThreads(void)       {return accelerator_threads;};
void     acceleratorThreads(uint32_t t) {accelerator_threads = t;};

void acceleratorInit(void)
{
  int nDevices = 1;
  cl::sycl::gpu_selector selector;
  cl::sycl::device selectedDevice { selector };
  theGridAccelerator = new sycl::queue (selectedDevice);

#ifdef GRID_SYCL_LEVEL_ZERO_IPC
  zeInit(0);
#endif
  
  char * localRankStr = NULL;
  int rank = 0, world_rank=0; 
#define ENV_LOCAL_RANK_OMPI    "OMPI_COMM_WORLD_LOCAL_RANK"
#define ENV_LOCAL_RANK_MVAPICH "MV2_COMM_WORLD_LOCAL_RANK"
#define ENV_RANK_OMPI          "OMPI_COMM_WORLD_RANK"
#define ENV_RANK_MVAPICH       "MV2_COMM_WORLD_RANK"
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK_OMPI)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_LOCAL_RANK_MVAPICH)) != NULL)
  {
    rank = atoi(localRankStr);		
  }
  if ((localRankStr = getenv(ENV_RANK_OMPI   )) != NULL) { world_rank = atoi(localRankStr);}
  if ((localRankStr = getenv(ENV_RANK_MVAPICH)) != NULL) { world_rank = atoi(localRankStr);}

  auto devices = cl::sycl::device::get_devices();
  for(int d = 0;d<devices.size();d++){

#define GPU_PROP_STR(prop) \
    printf("AcceleratorSyclInit:   " #prop ": %s \n",devices[d].get_info<cl::sycl::info::device::prop>().c_str());

#define GPU_PROP_FMT(prop,FMT) \
    printf("AcceleratorSyclInit:   " #prop ": " FMT" \n",devices[d].get_info<cl::sycl::info::device::prop>());

#define GPU_PROP(prop)             GPU_PROP_FMT(prop,"%ld");

    GPU_PROP_STR(vendor);
    GPU_PROP_STR(version);
    GPU_PROP(global_mem_size);

  }
  if ( world_rank == 0 ) {
    auto name = theGridAccelerator->get_device().get_info<sycl::info::device::name>();
    printf("AcceleratorSyclInit: Selected device is %s\n",name.c_str());
    printf("AcceleratorSyclInit: ================================================\n");
  }
}

void sharedMemoryInit(MPI_Comm comm);
void sharedMemoryAllocate(size_t bytes);
void sharedMemoryTest(size_t bytes);

MPI_Comm communicator_world;

void mpiInit(int *argc,char ***argv)
{
  int flag;
  int provided;

  MPI_Init_thread(argc,argv,MPI_THREAD_MULTIPLE,&provided);

  // Never clean up as done once.
  MPI_Comm_dup (MPI_COMM_WORLD,&communicator_world);

  sharedMemoryInit(communicator_world);
  sharedMemoryAllocate(1024L*1024L*1024L);
}

std::vector<void *> WorldShmCommBufs;

MPI_Comm WorldComm;
int           WorldRank;
int           WorldSize;

MPI_Comm WorldShmComm;
int           WorldShmRank;
int           WorldShmSize;

int           WorldNodes;
int           WorldNode;

std::vector<int>  WorldShmRanks;

void sharedMemoryInit(MPI_Comm comm)
{
#define header "SharedMemoryMpi: "

  WorldComm = comm;
  MPI_Comm_rank(WorldComm,&WorldRank);
  MPI_Comm_size(WorldComm,&WorldSize);
  // WorldComm, WorldSize, WorldRank

  /////////////////////////////////////////////////////////////////////
  // Split into groups that can share memory
  /////////////////////////////////////////////////////////////////////
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,&WorldShmComm);

  MPI_Comm_rank(WorldShmComm     ,&WorldShmRank);
  MPI_Comm_size(WorldShmComm     ,&WorldShmSize);

  if ( WorldRank == 0) {
    std::cout << header " World communicator of size " <<WorldSize << std::endl;  
    std::cout << header " Node  communicator of size " <<WorldShmSize << std::endl;
  }
  // WorldShmComm, WorldShmSize, WorldShmRank

  // WorldNodes
  WorldNodes = WorldSize/WorldShmSize;
  assert( (WorldNodes * WorldShmSize) == WorldSize );

  /////////////////////////////////////////////////////////////////////
  // find world ranks in our SHM group (i.e. which ranks are on our node)
  /////////////////////////////////////////////////////////////////////
  MPI_Group WorldGroup, ShmGroup;
  MPI_Comm_group (WorldComm, &WorldGroup); 
  MPI_Comm_group (WorldShmComm, &ShmGroup);

  std::vector<int> world_ranks(WorldSize);   for(int r=0;r<WorldSize;r++) world_ranks[r]=r;

  WorldShmRanks.resize(WorldSize); 
  MPI_Group_translate_ranks (WorldGroup,WorldSize,&world_ranks[0],ShmGroup, &WorldShmRanks[0]); 

  ///////////////////////////////////////////////////////////////////
  // Identify who is in my group and nominate the leader
  ///////////////////////////////////////////////////////////////////
  int g=0;
  std::vector<int> MyGroup;
  MyGroup.resize(WorldShmSize);
  for(int rank=0;rank<WorldSize;rank++){
    if(WorldShmRanks[rank]!=MPI_UNDEFINED){
      assert(g<WorldShmSize);
      MyGroup[g++] = rank;
    }
  }
  
  std::sort(MyGroup.begin(),MyGroup.end(),std::less<int>());
  int myleader = MyGroup[0];
  
  std::vector<int> leaders_1hot(WorldSize,0);
  std::vector<int> leaders_group(WorldNodes,0);
  leaders_1hot [ myleader ] = 1;
    
  ///////////////////////////////////////////////////////////////////
  // global sum leaders over comm world
  ///////////////////////////////////////////////////////////////////
  int ierr=MPI_Allreduce(MPI_IN_PLACE,&leaders_1hot[0],WorldSize,MPI_INT,MPI_SUM,WorldComm);
  assert(ierr==0);

  ///////////////////////////////////////////////////////////////////
  // find the group leaders world rank
  ///////////////////////////////////////////////////////////////////
  int group=0;
  for(int l=0;l<WorldSize;l++){
    if(leaders_1hot[l]){
      leaders_group[group++] = l;
    }
  }

  ///////////////////////////////////////////////////////////////////
  // Identify the node of the group in which I (and my leader) live
  ///////////////////////////////////////////////////////////////////
  WorldNode=-1;
  for(int g=0;g<WorldNodes;g++){
    if (myleader == leaders_group[g]){
      WorldNode=g;
    }
  }
}
void sharedMemoryAllocate(size_t bytes)
{
  void * ShmCommBuf ; 

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // allocate the pointer array for shared windows for our group
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  MPI_Barrier(WorldShmComm);
  WorldShmCommBufs.resize(WorldShmSize);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Each MPI rank should allocate our own buffer
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  ShmCommBuf = acceleratorAllocDevice(bytes);
  if (ShmCommBuf == (void *)NULL ) {
    std::cerr << " SharedMemoryMPI.cc acceleratorAllocDevice failed NULL pointer for " << bytes<<" bytes " << std::endl;
    exit(EXIT_FAILURE);  
  }
  //  if ( WorldRank == 0 ){
  if ( 1 ){
    std::cout << WorldRank << header " SharedMemoryMPI.cc acceleratorAllocDevice "<< bytes 
	      << "bytes at "<< std::hex<< ShmCommBuf <<std::dec<<" for comms buffers " <<std::endl;
  }
  char value='a';
  theGridAccelerator->fill((void *)ShmCommBuf, value, bytes).wait();

  std::cout<< "Setting up IPC"<<std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Loop over ranks/gpu's on our node
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  for(int r=0;r<WorldShmSize;r++){

#ifdef GRID_SYCL_LEVEL_ZERO_IPC

  auto zeDevice    = cl::sycl::get_native<cl::sycl::backend::level_zero>(theGridAccelerator->get_device());
  auto zeContext   = cl::sycl::get_native<cl::sycl::backend::level_zero>(theGridAccelerator->get_context());
    //////////////////////////////////////////////////
    // If it is me, pass around the IPC access key
    //////////////////////////////////////////////////
    typedef struct { int fd; pid_t pid ; } clone_mem_t;
    ze_ipc_mem_handle_t handle;
    clone_mem_t what_intel_should_have_done;
    std::cout << " sizeof(ze_ipc_mem_handle_t) is " <<sizeof(ze_ipc_mem_handle_t)<<std::endl;
    if ( r==WorldShmRank ) { 
      auto err = zeMemGetIpcHandle(zeContext,ShmCommBuf,&handle);
      if ( err != ZE_RESULT_SUCCESS ) {
	std::cout << "SharedMemoryMPI.cc zeMemGetIpcHandle failed for rank "<<r<<" "<<std::hex<<err<<std::dec<<std::endl;
	exit(EXIT_FAILURE);
      } else {
	std::cout << "SharedMemoryMPI.cc zeMemGetIpcHandle succeeded for rank "<<r<<" "<<std::hex<<err<<std::dec<<std::endl;
      }
      memcpy((void *)&what_intel_should_have_done.fd,(void *)&handle,sizeof(int));
      what_intel_should_have_done.pid = getpid();
      std::cout<<"Allocated Ipc filedes pid is {"<< what_intel_should_have_done.fd <<","
	       <<                                   what_intel_should_have_done.pid<<"}\n";

      std::cout<<"Allocated IpcHandle rank size "<< sizeof(handle) << " rank "<<r<<" (hex) ";
      for(int c=0;c<ZE_MAX_IPC_HANDLE_SIZE;c++){
	std::cout<<std::hex<<(uint32_t)((uint8_t)handle.data[c])<<std::dec;
      }
      std::cout<<std::endl;
    }
    //////////////////////////////////////////////////
    // Share this IPC handle across the Shm Comm
    //////////////////////////////////////////////////
    { 
      int ierr=MPI_Bcast(&what_intel_should_have_done,
			 sizeof(what_intel_should_have_done),
			 MPI_BYTE,
			 r,
			 WorldShmComm);
      assert(ierr==0);
    }
    
    ///////////////////////////////////////////////////////////////
    // If I am not the source, overwrite thisBuf with remote buffer
    ///////////////////////////////////////////////////////////////
    void * thisBuf = ShmCommBuf;
    if ( r!=WorldShmRank ) {
      thisBuf = nullptr;
      std::cout<<"mapping seeking remote pid/fd "
	       <<what_intel_should_have_done.pid<<"/"
	       <<what_intel_should_have_done.fd<<std::endl;

      int pidfd = syscall(SYS_pidfd_open,what_intel_should_have_done.pid,0);
      std::cout<<"Using IpcHandle pidfd "<<pidfd<<"\n";
      //      int myfd  = syscall(SYS_pidfd_getfd,pidfd,what_intel_should_have_done.fd,0);
      int myfd  = syscall(438,pidfd,what_intel_should_have_done.fd,0);

      std::cout<<"Using IpcHandle myfd "<<myfd<<"\n";
      
      memcpy((void *)&handle,(void *)&myfd,sizeof(int));

      std::cout<<"Using IpcHandle rank "<<r<<" ";
      for(int c=0;c<ZE_MAX_IPC_HANDLE_SIZE;c++){
	std::cout<<std::hex<<(uint32_t)((uint8_t)handle.data[c])<<std::dec;
      }
      std::cout<<std::endl;
      auto err = zeMemOpenIpcHandle(zeContext,zeDevice,handle,0,&thisBuf);
      if ( err != ZE_RESULT_SUCCESS ) {
	std::cout << "SharedMemoryMPI.cc "<<zeContext<<" "<<zeDevice<<std::endl;
	std::cout << "SharedMemoryMPI.cc zeMemOpenIpcHandle failed for rank "<<r<<" "<<std::hex<<err<<std::dec<<std::endl; 
	exit(EXIT_FAILURE);
      } else {
	std::cout << "SharedMemoryMPI.cc zeMemOpenIpcHandle succeeded for rank "<<r<<std::endl;
	std::cout << "SharedMemoryMPI.cc zeMemOpenIpcHandle pointer is "<<std::hex<<thisBuf<<std::dec<<std::endl;
      }
      assert(thisBuf!=nullptr);
    }
    ///////////////////////////////////////////////////////////////
    // Save a copy of the device buffers
    ///////////////////////////////////////////////////////////////
    WorldShmCommBufs[r] = thisBuf;

#else
    WorldShmCommBufs[r] = ShmCommBuf;
#endif
  }

}
#undef header
void sharedMemoryTest(size_t bytes)
{
  double elapsed;
  double fill;
  if ( WorldRank==0){

    std::cout << "SharedMemoryTest for "<<bytes<<"bytes"<<std::endl;

    for(int r=0;r<WorldShmSize;r++){

      uint64_t word = (0x5A5AUL<<32) | r;
      uint64_t words = bytes/sizeof(uint64_t);

      uint64_t *from = (uint64_t *)WorldShmCommBufs[r];
      uint64_t *to   = (uint64_t *)WorldShmCommBufs[0];
      fill=-usecond();
      accelerator_for(w,words,1,{
	  from[w] = word;
	});
      fill+=usecond();

      elapsed=-usecond();
      accelerator_for(w,words,1,{
	  to[w]=from[w];
	});
      elapsed+=usecond();

      std::cout << "Get from device "<<r<<" to device 0 " << " transfer "<<bytes <<" bytes in " <<elapsed <<" us\n";
      std::cout << "Get from device "<<r<<" to device 0 " << " rate "<< bytes/elapsed<< " MB/s\n";
      std::cout << "Device "<<r<<" from device 0 " << " fill "<< bytes/fill<< " MB/s\n";

      to   = (uint64_t *)WorldShmCommBufs[r];
      from = (uint64_t *)WorldShmCommBufs[0];
      fill=-usecond();
      accelerator_for(w,words,1,{
	  from[w] = word;
	});
      fill+=usecond();
      elapsed=-usecond();
      accelerator_for(w,words,1,{
	  to[w]=from[w];
	});
      elapsed+=usecond();

      std::cout << "Put to device "<<r<<" from device 0 " << " transfer "<<bytes <<" bytes in " <<elapsed <<" us\n";
      std::cout << "Put to device "<<r<<" from device 0 " << " rate "<< bytes/elapsed<< " MB/s\n";
      std::cout << "Device 0 from device 0 " << " fill "<< bytes/fill<< " MB/s\n";
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc,char **argv)
{
  acceleratorInit();

  mpiInit(&argc,&argv);

  sharedMemoryTest(1024*1024);
  sharedMemoryTest(1024*1024*2);
  sharedMemoryTest(1024*1024*4);
  sharedMemoryTest(1024*1024*8);
  sharedMemoryTest(1024*1024*16);
  sharedMemoryTest(1024*1024*32);
  sharedMemoryTest(1024*1024*32);
  sharedMemoryTest(1024*1024*64);
  sharedMemoryTest(1024*1024*128);
  sharedMemoryTest(1024*1024*256);
  sharedMemoryTest(1024*1024*512);
  
}
