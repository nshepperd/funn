{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
module AI.Funn.Diff.Diff (
  Additive(..), (##), unit,
  Derivable(..),
  Diff(..),
  runDiff_, runDiffForward,
  first, second, (>>>),
  assocL, assocR, swap,
  fst, snd
  ) where

import           Prelude hiding ((.), id, fst, snd)

import           Control.Applicative
import           Control.Category
import           Control.Monad
import           Data.Foldable
import           Data.Functor.Identity
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V

class Derivable a where
  type family D a :: *

-- Diff represents the type of differentiable functions
newtype Diff m a b = Diff {
  runDiff :: a -> m (b, D b -> m (D a))
  }

class Additive m a where
  plus :: a -> a -> m a
  zero :: m a
  plusm :: (Foldable f) => f a -> m a
  default plusm :: (Monad m, Foldable f) => f a -> m a
  plusm xs = do z <- zero
                foldrM plus z xs

(##) :: (Additive Identity a) => a -> a -> a
x ## y = runIdentity (plus x y)

unit :: (Additive Identity a) => a
unit = runIdentity zero

instance Derivable Double where
  type D Double = Double

instance Derivable () where
  type D () = ()

instance Derivable Int where
  type D Int = ()

instance (Derivable a, Derivable b) => Derivable (a, b) where
  type D (a, b) = (D a, D b)

instance Derivable a => Derivable (Vector a) where
  type D (Vector a) = Vector (D a)

instance (Applicative m) => Additive m () where
  plus _ _ = pure ()
  zero = pure ()
  plusm _ = pure ()

instance (Applicative m, Additive m a, Additive m b) => Additive m (a, b) where
  plus (a1, b1) (a2, b2) = liftA2 (,) (plus a1 a2) (plus b1 b2)
  zero = liftA2 (,) zero zero
  plusm abs = let (as, bs) = unzip (toList abs)
              in liftA2 (,) (plusm as) (plusm bs)

instance (Applicative m) => Additive m Double where
  plus a b = pure (a + b)
  zero = pure 0
  plusm xs = pure (sum xs)

runDiff_ :: Diff Identity a b -> a -> (b, D b -> D a)
runDiff_ (Diff f) a = let (b, dba) = runIdentity (f a) in
                       (b, runIdentity . dba)

runDiffForward :: Monad m => Diff m a b -> a -> m b
runDiffForward d a = do (b, _) <- runDiff d a
                        return b

diff_id :: (Applicative m) => Diff m a a
diff_id = Diff (\a -> pure (a, \db -> pure db))

diff_connect :: (Monad m) => Diff m a b -> Diff m b c -> Diff m a c
diff_connect mab mbc = Diff run
  where
    run a = do (b, dba) <- runDiff mab a
               (c, dcb) <- runDiff mbc b
               return (c, dcb >=> dba)

instance Monad m => Category (Diff m) where
  id = diff_id
  (.) = flip diff_connect

first :: (Monad m) => Diff m a b -> Diff m (a,c) (b,c)
first net = Diff run
  where
    run (a, c) = do (b, dba) <- runDiff net a
                    let back (db, dc) = do
                          da <- dba db
                          return (da, dc)
                    return ((b, c), back)

second :: (Monad m) => Diff m a b -> Diff m (c,a) (c,b)
second net = Diff run
  where
    run (c, a) = do (b, dba) <- runDiff net a
                    let back (dc, db) = do
                          da <- dba db
                          return (dc, da)
                    return ((c, b), back)

assocL :: (Applicative m) => Diff m (a,(b,c)) ((a,b),c)
assocL = Diff run
  where
    run (a,(b,c)) = let back ((da,db),dc) = pure (da,(db,dc)) in
                     pure (((a,b),c), back)

assocR :: (Applicative m) => Diff m ((a,b),c) (a,(b,c))
assocR = Diff run
  where
    run ((a,b),c) = let back (da,(db,dc)) = pure ((da,db),dc) in
                     pure ((a,(b,c)), back)

swap :: (Applicative m) => Diff m (a,b) (b,a)
swap = Diff run
  where
    run (a,b) = pure ((b,a), \(db,da) -> pure (da,db))

fst :: (Applicative m, Additive m (D b)) => Diff m (a,b) a
fst = Diff run
  where
    run (a,_) = let back da = fmap ((,) da) zero
                in pure (a, back)

snd :: (Applicative m, Additive m (D a)) => Diff m (a,b) b
snd = Diff run
  where
    run (_,b) = let back db = fmap (\da -> (da, db)) zero
                in pure (b, back)
