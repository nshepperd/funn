{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.TensorLazy (
  Tensor(..),
  -- Core
  fromStrict,
  toStrict,
  append,
  nul
  ) where


import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.LazyMem (LazyMem)
import qualified AI.Funn.CL.LazyMem as LazyMem
import           AI.Funn.CL.MonadCL
import qualified AI.Funn.CL.Tensor as T
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space

newtype Tensor (ds :: [Nat]) = Tensor { getTensor :: LazyMem Double }

instance (MonadIO m, KnownDims ds) => Zero m (Tensor ds) where
  zero = fromStrict <$> zero

instance (MonadIO m, KnownDims ds) => Semi m (Tensor ds) where
  -- Expensive.
  plus a b = fromStrict <$> plus (toStrict a) (toStrict b)

instance (MonadIO m, KnownDims ds) => Additive m (Tensor ds) where
  -- Expensive.
  plusm xs = fromStrict <$> plusm (map toStrict (toList xs))

instance (MonadIO m, KnownDims ds) => Scale m Double (Tensor ds) where
  -- Expensive
  scale x xs = fromStrict <$> scale x (toStrict xs)

instance (MonadIO m, KnownDims ds) => VectorSpace m Double (Tensor ds) where
  {}

instance (MonadIO m, KnownDims ds) => Inner m Double (Tensor ds) where
  -- Expensive.
  inner x y = inner (toStrict x) (toStrict y)

instance (MonadIO m, KnownDims ds) => Finite m Double (Tensor ds) where
  -- Expensive.
  getBasis b = getBasis (toStrict b)

-- O(1)
fromStrict :: T.Tensor ds -> Tensor ds
fromStrict (T.Tensor mem) = Tensor (LazyMem.fromStrict mem)

-- up to O(size)
toStrict :: Tensor ds -> T.Tensor ds
toStrict (Tensor buf) = T.Tensor (unsafePerformIO $ LazyMem.toStrict buf)

-- O(1)
append :: Tensor '[a] -> Tensor '[b] -> Tensor '[a+b]
append (Tensor one) (Tensor two) = Tensor (LazyMem.append one two)

nul :: Tensor '[0]
nul = Tensor mempty
