{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.Batched.BTensor (
  BTensor(..)
  ) where


import           Control.Applicative
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import qualified AI.Funn.CL.LazyMem as LM
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Space

newtype BTensor (ω :: Nat) (ds :: [Nat]) = BTensor { getBTensor :: Tensor ds }

instance Derivable (BTensor ω ds) where
  type D (BTensor ω ds) = Tensor (ω ': ds)

instance (MonadIO m, KnownDims ds) => Zero m (BTensor ω ds) where
  zero = BTensor <$> zero

instance (MonadIO m, KnownDims ds) => Semi m (BTensor ω ds) where
  plus (BTensor x) (BTensor y) = BTensor <$> plus x y

instance (MonadIO m, KnownDims ds) => Additive m (BTensor ω ds) where
  plusm xs = BTensor <$> plusm (map getBTensor xs)

instance (MonadIO m, KnownDims ds) => Scale m Double (BTensor ω ds) where
  scale x (BTensor xs) = BTensor <$> scale x xs

instance (MonadIO m, KnownDims ds) => VectorSpace m Double (BTensor ω ds) where
  {}

instance (MonadIO m, KnownDims ds) => Inner m Double (BTensor ω ds) where
  inner (BTensor x) (BTensor y) = inner x y

instance (MonadIO m, KnownDims ds) => Finite m Double (BTensor ω ds) where
  getBasis (BTensor x) = getBasis x
