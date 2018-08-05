{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeOperators #-}
module AI.Funn.Indexed.Indexed where

import GHC.TypeLits

class Indexed (cat :: Nat -> k -> k -> *) where
  iid :: cat 0 a a
  (~>>) :: (KnownNat i, KnownNat j) => cat i a b -> cat j b c -> cat (i + j) a c

class Indexed cat => Assoc (cat :: Nat -> * -> * -> *) where
  first :: cat i a b -> cat i (a, c) (b, c)
  second :: cat i a b -> cat i (c, a) (c, b)
  swap :: cat 0 (a, b) (b, a)
  assocR :: cat 0 ((a,b),c) (a,(b,c))
  assocL :: cat 0 (a,(b,c)) ((a,b),c)

(***) :: (KnownNat i, KnownNat j, Assoc cat) => cat i a b -> cat j c d -> cat (i + j) (a,c) (b,d)
(***) one two = first one ~>> second two
