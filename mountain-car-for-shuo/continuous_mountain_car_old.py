    # OLD monitoring goals
        self.MONINTC = [-0.1, 0.1] 
        self.MONINTD = [-0.01, 0.01] 
        self.MONINTINITPOS = [-0.53, -0.48]
        # exact encoding of the successful verifications


# very initial version, for small c
        self.VERIFASSN1 = lambda c, d, i : \
            -.1 <= c <= .1 and \
            ( 
                -.01 <= d <= -.0075 and -.56 <= i <= -.41 or \
                -.0075 <= d <= -.005 and -.57 <= i <= -.42 or \
                -.005 <= d <= -.0025 and (-.58 <= i <= -.42 or -.41 <= i <= -.4) or \
                -.0025 <= d <= .0025 and (-.58 <= i <= -.43 or -.41 <= i <= -.4) or \
                .0025 <= d <= .005 and (-.56 <= i <= -.43 or -.41 <= i <= -.4) or \
                .005 <= d <= .0075 and (-.55 <= i <= -.43 or -.42 <= i <= -.4) or \
                .0075 <= d <= .01 and (-.53 <= i <= -.43 or -.42 <= i <= -.4)
            )
        # with fake yellow results
        self.VERIFASSN2 = lambda c,d,i: \
                -.01 <= d <= -.0075 and ( \
                -.59 <= i <= -.58 and (-.9 <= c <= -.8 or  0 <= c <= .1) or \
                -.58 <= i <= -.57 and (-.6 <= c <= -.5 or .5 <= c <= .6) or \
                -.56 <= i <= -.41 and (-1 <= c <= 1) \
                ) or \
                -.0075 <= d <= -.005 and ( \
                -.57 <= i <= -.42 and -1 <= c <= 1 \
                ) or \
                -.005 <= d <= -.0025 and ( \
                -.58 <= i <= -.42 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or  \
                -.0025 <= d <= 0 and ( \
                -.59 <= i <= -.58 and .2 <= c <= 1 or \
                -.58 <= i <= -.43 and -1 <= c <= 1 or \
                -.42 <= i <= -.41 and .5 <= c <= .6 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                0 <= d <= .0025 and ( \
                -.58 <= i <= -.57 and -.4 <= c <= 1 or \
                -.57 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0025 <= d <= .005 and ( \
                -.57 <= i <= -.56 and .1 <= c <= 1 or \
                -.56 <= i <= -.43 and -1 <= c <= 1 \
                ) or \
                .005 <= d <= .0075 and ( \
                -.59 <= i <= -.58 and -1 <= c <= -.9 or \
                -.55 <= i <= -.54 and -.9 <= c <= 1 or \
                -.54 <= i <= -.43 and -1 <= c <= 1 or \
                -.42 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                 .0075 <= d <= .01 and ( \
                -.59 <= i <= -.58 and (0 <= c <= .1 or .3 <= c <= .4 or .7 <= c <= .8) or \
                -.58 <= i <= -.57 and (-.9 <= c <= -.7) or \
                -.53 <= i <= -.43 and (-1 <= c <= 1) or \
                -.42 <= i <= -.4 and (-1 <= c <= 1) \
                ) 
        # without fake "yellow" results
        self.VERIFASSN3 = lambda c,d,i: \
                -.01 <= d <= -.0075 and ( \
                -.56 <= i <= -.41 and (-1 <= c <= 1) \
                ) or \
                -.0075 <= d <= -.005 and ( \
                -.57 <= i <= -.42 and -1 <= c <= 1 \
                ) or \
                -.005 <= d <= -.0025 and ( \
                -.58 <= i <= -.42 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or  \
                -.0025 <= d <= 0 and ( \
                -.59 <= i <= -.58 and .2 <= c <= 1 or \
                -.58 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                0 <= d <= .0025 and ( \
                -.58 <= i <= -.57 and -.4 <= c <= 1 or \
                -.57 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0025 <= d <= .005 and ( \
                -.57 <= i <= -.56 and .1 <= c <= 1 or \
                -.56 <= i <= -.43 and -1 <= c <= 1 \
                ) or \
                .005 <= d <= .0075 and ( \
                -.55 <= i <= -.54 and -.9 <= c <= 1 or \
                -.54 <= i <= -.43 and -1 <= c <= 1 or \
                -.42 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                 .0075 <= d <= .01 and ( \
                -.53 <= i <= -.43 and (-1 <= c <= 1) or \
                -.42 <= i <= -.4 and (-1 <= c <= 1) \
                ) 



