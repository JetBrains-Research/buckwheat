<?php
// read file to contents variable
$contents = file_get_contents( dirname( __FILE__ ) . "/simple.set" );
$set = QuickHashIntSet::loadFromString(
    $contents,
    QuickHashIntSet::DO_NOT_USE_ZEND_ALLOC
);

/*Print information using foreach loop.
On each iteration check set for a key*/
foreach( range( 0, 0x0f ) as $key )
{
    printf( "Key %3d (%2x) is %s\n",
        $key, $key, 
        $set->exists( $key ) ? 'set' : 'unset'
    );
}
?>
