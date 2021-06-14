#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 27/11/2020
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class DawisError(Exception):
    """Base class for errors in DAWIS.
    """
    def __repr__(self):
        return 'dawis.DawisError(%r)' %(str(self))


class DawisDimensionError(DawisError):
    """A DAWIS-specific error class raised when the number of dimensions of an
    input array does not match the number of dimension required by the algorithm.
    """
    def __init__(self, message, ndim, allowed_ndim):
        self.message      = message
        self.ndim         = ndim
        self.allowed_ndim = allowed_ndim

        message = ''.join( ( self.message, \
                             ' number of dimensions is ', \
                             str(ndim), \
                             ' instead of ', \
                             str(allowed_ndim) ) )
        super().__init__(message)

    def __repr__(self):
        return 'dawis.DawisDimensionError(%r)' %(str(self))


class DawisUnknownType(DawisError):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message, type):
        self.message = message
        self.type = type

        message = ''.join( ( self.message, \
                             ' type ', \
                             str(type), \
                             ' is unknown.' ))
        super().__init__(message)


class DawisWrongType(DawisError):
    """A DAWIS-specific error class raised when a wrong argument type is given
    to a function or a class.
    """

    def __init__(self, message, type_error, allowed_type):
        self.message = message
        self.type_error = type_error
        self.allowed_type = allowed_type

        message = ''.join( ( self.message, \
                             ' type is ', \
                             str(type_error), \
                             ' instead of ', \
                             str(allowed_type) ) )
        super().__init__(message)

    def __repr__(self):
        return 'dawis.DawisDimensionError(%r)' %(str(self))

class DawisValueError(DawisError):
    """A DAWIS-specific error class raised when a wrong argument value is given
    to a function or a class.
    """

    def __init__(self, message, value_error, allowed_value):
        self.message = message
        self.value_error = value_error
        self.allowed_value = allowed_value

        message = ''.join( ( self.message, \
                             ' value ', \
                             str(value_error), \
                             ' is not allowed, must be ', \
                             str(allowed_value) ) )
        super().__init__(message)

    def __repr__(self):
        return 'dawis.DawisDimensionError(%r)' %(str(self))

if __name__ == '__main__':

    raise DawisDimensionError('Not a datacube -->', 2, 3)
    raise DawisDimensionError('Not an image -->', 3, 2)
