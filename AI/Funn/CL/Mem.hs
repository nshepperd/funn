{-# LANGUAGE ScopedTypeVariables, ForeignFunctionInterface #-}
module AI.Funn.CL.Mem (
  Mem, malloc, free, arg,
  fromList, toList
  ) where

import           Control.Exception
import           Control.Monad.IO.Class
import           Data.Int
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           Foreign.Storable (Storable(..))
import           System.Mem
import           Unsafe.Coerce

import qualified Foreign.OpenCL.Bindings as CL
import qualified Foreign.OpenCL.Bindings.Synchronization as CL
import qualified Foreign.OpenCL.Bindings.Types as CL

import           AI.Funn.CL.MonadCL

data CMem
newtype Mem s a = Mem (ForeignPtr CMem)
                deriving Show

foreign import ccall "&clmem_free" clmem_free :: FunPtr (Ptr CMem -> IO ())
foreign import ccall "clmem_increase" clmem_increase :: Int64 -> IO ()
foreign import ccall "clmem_count" clmem_count :: IO Int64

malloc :: (MonadCL s m, Storable a) => Int -> m (Mem s a)
malloc n = do ctx <- getContext
              fromAlloc (CL.mallocArray ctx [] n)

free :: MonadIO m => Mem s a -> m ()
free (Mem foreignptr) = liftIO (finalizeForeignPtr foreignptr)

arg :: Mem s a -> KernelArg s
arg mem = KernelArg run
  where
    run k = withMem mem (k . pure . CL.MObjArg)

fromList :: (MonadCL s m, Storable a) => [a] -> m (Mem s a)
fromList xs = do ctx <- getContext
                 fromAlloc (CL.newListArray ctx xs)

toList :: forall a m s. (MonadCL s m, Storable a) => Mem s a -> m [a]
toList mem = do queue <- getCommandQueue
                liftIO $ tryAllocation $ withMem mem $ \memobj -> do
                  n <- (`div` elemSize) . fromIntegral <$> CL.memobjSize memobj
                  CL.peekListArray queue n memobj
  where
    elemSize :: Int
    elemSize = sizeOf (undefined :: a)

-- Utility functions --

fromAlloc :: MonadCL s m => IO (CL.MemObject a) -> m (Mem s a)
fromAlloc alloc = liftIO $ do memobj <- tryAllocation alloc
                              size <- CL.memobjSize memobj
                              clmem_increase (fromIntegral size)
                              foreignptr <- newForeignPtr clmem_free (unsafeExtract memobj)
                              return (Mem foreignptr)

tryAllocation :: IO a -> IO a
tryAllocation m = catch m (\(e :: CL.ClException) -> tryAgain)
  where
    tryAgain = do
      -- try again after hopefully freeing space
      putStrLn "Try Minor GC"
      performMinorGC
      catch m (\(e :: CL.ClException) -> tryMajor)

    tryMajor = do
      putStrLn "Try Major GC"
      performGC
      m

withMem :: Mem s a -> (CL.MemObject a -> IO r) -> IO r
withMem (Mem foreignptr) k = withForeignPtr foreignptr (k . unsafeInject)

-- Gets the Ptr out of hopencl.

data Hack = Hack (Ptr CMem)
unsafeExtract :: CL.MemObject a -> Ptr CMem
unsafeExtract memobj = case unsafeCoerce memobj of
                        Hack r -> r

unsafeInject :: Ptr CMem -> CL.MemObject a
unsafeInject ptr = unsafeCoerce (Hack ptr)
