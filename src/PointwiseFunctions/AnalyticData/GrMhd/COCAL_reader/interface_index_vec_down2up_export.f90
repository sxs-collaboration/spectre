module interface_index_vec_down2up_export
  implicit none
  interface 
    subroutine index_vec_down2up_export(hxxu,hxyu,hxzu,hyyu,hyzu,hzzu,vecxu,vecyu,veczu,vecxd,vecyd,veczd)
      real(8), pointer :: vecxu(:,:,:), vecyu(:,:,:), veczu(:,:,:), &
          &               vecxd(:,:,:), vecyd(:,:,:), veczd(:,:,:), &
          &               hxxu(:,:,:),  hxyu(:,:,:),  hxzu(:,:,:), &
          &               hyyu(:,:,:),  hyzu(:,:,:),  hzzu(:,:,:)
    end subroutine index_vec_down2up_export
  end interface
end module interface_index_vec_down2up_export
