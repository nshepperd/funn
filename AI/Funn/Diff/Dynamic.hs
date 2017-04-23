{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs, RankNTypes, ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances, DataKinds #-}
module AI.Funn.Diff.Dynamic where

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.Monoid

import           Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IntMap

import           Unsafe.Coerce
import           AI.Funn.Diff.Diff (Additive(..), Derivable(..))

newtype Ref s a = Ref { getRef :: Int }

data Dynamic where
  Dynamic :: a -> Dynamic
unsafe :: Dynamic -> a
unsafe (Dynamic a) = unsafeCoerce a

newtype DiffMap s = DiffMap (IntMap Dynamic)
newtype DiffMapD s = DiffMapD (IntMap Dynamic)

instance Derivable (DiffMap s) where
  type D (DiffMap s) = DiffMapD s

emptyMap :: DiffMap s
emptyMap = DiffMap (IntMap.empty)

readMap :: Ref s a -> DiffMap s -> a
readMap (Ref ref) (DiffMap m) = unsafe (m IntMap.! ref)

writeMap :: Ref s a -> Maybe a -> DiffMap s -> DiffMap s
writeMap (Ref ref) (Just a) (DiffMap m) = DiffMap (IntMap.insert ref (Dynamic a) m)
writeMap (Ref ref) Nothing (DiffMap m) = DiffMap (IntMap.delete ref m)


emptyMapD :: DiffMapD s
emptyMapD = DiffMapD (IntMap.empty)

readMapD :: (Derivable a) => Ref s a -> DiffMapD s -> Maybe (D a)
readMapD (Ref ref) (DiffMapD m) = unsafe <$> (IntMap.lookup ref m)

readMapDA :: (Applicative m, Derivable a, Additive m (D a)) => Ref s a -> DiffMapD s -> m (D a)
readMapDA ref m = case readMapD ref m of
                    Just da -> pure da
                    Nothing -> zero

writeMapD :: (Derivable a) => Ref s a -> Maybe (D a) -> DiffMapD s -> DiffMapD s
writeMapD (Ref ref) (Just a) (DiffMapD m) = DiffMapD (IntMap.insert ref (Dynamic a) m)
writeMapD (Ref ref) Nothing (DiffMapD m) = DiffMapD (IntMap.delete ref m)

addMapD :: (Applicative m, Additive m (D a), Derivable a) => Ref s a -> Maybe (D a) -> DiffMapD s -> m (DiffMapD s)
addMapD ref (Just a) m = case readMapD ref m of
                           Just a1 -> (\new_a -> writeMapD ref (Just new_a) m) <$> plus a a1
                           Nothing -> pure (writeMapD ref (Just a) m)
addMapD ref Nothing m = pure m
